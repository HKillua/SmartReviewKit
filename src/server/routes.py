"""FastAPI route definitions."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.server.models import ChatRequest, HealthResponse, UploadResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_chat_handler = None
_agent = None
_upload_dir = "data/uploads"
_max_upload_mb = 50


def configure_routes(chat_handler: Any, agent: Any, upload_dir: str = "data/uploads", max_upload_mb: int = 50) -> None:
    global _chat_handler, _agent, _upload_dir, _max_upload_mb
    _chat_handler = chat_handler
    _agent = agent
    _upload_dir = upload_dir
    _max_upload_mb = max_upload_mb


@router.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """SSE streaming chat endpoint."""
    if _chat_handler is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    from sse_starlette.sse import EventSourceResponse

    async def event_generator():
        async for chunk in _chat_handler.handle_stream(request):
            yield {"event": "message", "data": chunk.model_dump_json()}

    return EventSourceResponse(event_generator())


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

    save_dir = Path(_upload_dir) / user_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename
    save_path.write_bytes(content)

    if _agent is None:
        return UploadResponse(success=True, filename=file.filename)

    from src.agent.types import ToolCallData, ToolContext
    tool_ctx = ToolContext(user_id=user_id, conversation_id="upload")
    tool_call = ToolCallData(id="upload", name="document_ingest", arguments={"file_path": str(save_path), "collection": collection})
    result = await _agent.tools.execute(tool_call, tool_ctx)

    if result.success:
        return UploadResponse(
            success=True,
            filename=file.filename,
            doc_id=result.metadata.get("doc_id", ""),
            chunk_count=result.metadata.get("chunk_count", 0),
            image_count=result.metadata.get("image_count", 0),
        )
    return UploadResponse(success=False, filename=file.filename, error=result.error)


@router.get("/api/health", response_model=HealthResponse)
async def health():
    tools_count = len(_agent.tools.tool_names) if _agent else 0
    return HealthResponse(tools_registered=tools_count)
