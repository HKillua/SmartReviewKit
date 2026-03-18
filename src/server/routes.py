"""FastAPI route definitions."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response

from src.server.models import ChatRequest, HealthResponse, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_chat_handler = None
_agent = None
_upload_dir = "data/uploads"
_max_upload_mb = 50
_feedback_store = None
_object_store = None
_task_store = None


def configure_routes(
    chat_handler: Any,
    agent: Any,
    upload_dir: str = "data/uploads",
    max_upload_mb: int = 50,
    feedback_store: Any = None,
    object_store: Any = None,
    task_store: Any = None,
) -> None:
    global _chat_handler, _agent, _upload_dir, _max_upload_mb, _feedback_store, _object_store, _task_store
    _chat_handler = chat_handler
    _agent = agent
    _upload_dir = upload_dir
    _max_upload_mb = max_upload_mb
    _feedback_store = feedback_store
    _object_store = object_store
    _task_store = task_store


@router.post("/api/chat")
async def chat_endpoint(http_request: Request, request: ChatRequest):
    """SSE streaming chat endpoint."""
    if _chat_handler is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        async for chunk in _chat_handler.handle_stream(request):
            if await http_request.is_disconnected():
                logger.info("Client disconnected, stopping SSE stream")
                break
            yield {"event": "message", "data": chunk.model_dump_json()}

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


@router.get("/api/conversations")
async def list_conversations(user_id: str = "default_user", limit: int = 50):
    """List conversations for a user (for sidebar)."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    convs = await _agent.conversations.list_conversations(user_id, limit=limit)
    return [
        {
            "id": c.id,
            "title": c.title or c.messages[0].content[:30] + "..." if c.messages else "新对话",
            "updated_at": c.updated_at.isoformat(),
            "message_count": len(c.messages),
        }
        for c in convs
    ]


@router.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user_id: str = "default_user"):
    """Retrieve conversation history."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    conv = await _agent.conversations.get(conversation_id, user_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv.model_dump()


@router.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str = "default_user"):
    """Delete a conversation."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    deleted = await _agent.conversations.delete(conversation_id, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True}


@router.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), user_id: str = "default_user", collection: str = "default"):
    """Upload a PDF or PPTX file and ingest into knowledge base."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".pptx"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    content = await file.read()
    if len(content) > _max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {_max_upload_mb}MB)")

    safe_filename = Path(file.filename).name
    save_dir = Path(_upload_dir) / user_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = (save_dir / safe_filename).resolve()
    if not str(save_path).startswith(str(save_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    save_path.write_bytes(content)
    uploaded_object_key = ""
    if _object_store is not None:
        uploaded_object_key = f"uploads/{user_id}/{uuid.uuid4().hex[:12]}_{safe_filename}"
        try:
            _object_store.put_bytes(uploaded_object_key, content)
        except Exception:
            logger.warning("Failed to upload file to object store", exc_info=True)
            uploaded_object_key = ""

    if _agent is None:
        return UploadResponse(success=True, filename=file.filename)

    from src.agent.types import ToolCallData, ToolContext
    tool_ctx = ToolContext(
        user_id=user_id,
        conversation_id="upload",
        metadata={"uploaded_object_key": uploaded_object_key} if uploaded_object_key else {},
    )
    tool_call = ToolCallData(id="upload", name="document_ingest", arguments={"file_path": str(save_path), "collection": collection})
    result = await _agent.tools.execute(tool_call, tool_ctx)

    if result.success:
        return UploadResponse(
            success=True,
            filename=file.filename,
            doc_id=result.metadata.get("doc_id", ""),
            chunk_count=result.metadata.get("chunk_count", 0),
            image_count=result.metadata.get("image_count", 0),
            task_id=result.metadata.get("task_id", ""),
            status=result.metadata.get("status", ""),
        )
    return UploadResponse(success=False, filename=file.filename, error=result.error)


@router.get("/api/ingestion/tasks/{task_id}")
async def get_ingestion_task(task_id: str):
    if _task_store is None:
        raise HTTPException(status_code=503, detail="Ingestion task store not configured")
    task = _task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/api/ingestion/tasks")
async def list_ingestion_tasks(user_id: str = "default_user", status: str = "", limit: int = 20):
    if _task_store is None:
        raise HTTPException(status_code=503, detail="Ingestion task store not configured")
    return _task_store.list_tasks(user_id=user_id, status=status or None, limit=limit)


@router.post("/api/feedback")
async def submit_feedback(
    user_id: str = "default_user",
    conversation_id: str = "",
    rating: str = "up",
    message_index: int = -1,
    comment: str = "",
    query: str = "",
    response_preview: str = "",
):
    """Submit user feedback (thumbs up/down) on a response."""
    if _feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not configured")
    if rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    fb_id = await _feedback_store.add(
        user_id=user_id,
        conversation_id=conversation_id,
        rating=rating,
        message_index=message_index,
        comment=comment,
        query=query,
        response_preview=response_preview[:200],
    )
    return {"success": True, "feedback_id": fb_id}


@router.get("/api/feedback/stats")
async def feedback_stats():
    """Get aggregate feedback statistics."""
    if _feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not configured")
    return await _feedback_store.stats()


@router.get("/api/health", response_model=HealthResponse)
async def health():
    tools_count = len(_agent.tools.tool_names) if _agent else 0
    return HealthResponse(tools_registered=tools_count)


@router.get("/api/artifacts/{artifact_id}")
async def download_artifact(artifact_id: str, user_id: str = "default_user", conversation_id: str = ""):
    """Download a generated learning artifact from object storage."""
    if _object_store is None:
        raise HTTPException(status_code=503, detail="Artifact store not configured")
    safe_name = Path(artifact_id).name
    if safe_name != artifact_id:
        raise HTTPException(status_code=400, detail="Invalid artifact id")
    convo = conversation_id.strip()
    if not convo:
        raise HTTPException(status_code=400, detail="conversation_id is required")
    object_key = f"artifacts/{convo}/{safe_name}"
    try:
        content = await asyncio.to_thread(_object_store.read_bytes, object_key)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None
    except Exception:
        logger.warning("Failed to read artifact %s", object_key, exc_info=True)
        raise HTTPException(status_code=404, detail="Artifact not found") from None

    media_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    headers = {
        "Content-Disposition": f'attachment; filename="{safe_name}"',
        "X-Artifact-User": user_id,
    }
    return Response(content=content, media_type=media_type, headers=headers)
