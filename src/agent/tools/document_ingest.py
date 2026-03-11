"""Document ingestion tool — triggers PDF/PPT ingestion into the knowledge base."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.agent.tools.base import Tool
from src.agent.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class DocumentIngestArgs(BaseModel):
    file_path: str = Field(..., description="待入库文件的路径")
    collection: str = Field(default="computer_network", description="目标 collection 名称，必须使用英文")


class DocumentIngestTool(Tool[DocumentIngestArgs]):
    """Ingest a PDF or PPTX file into the knowledge base via IngestionPipeline."""

    def __init__(
        self,
        settings: Any = None,
        pipeline: Any = None,
        allowed_dirs: list[str] | None = None,
        semantic_cache: Any = None,
        object_store: Any = None,
        document_registry: Any = None,
        task_store: Any = None,
    ) -> None:
        self._settings = settings
        self._pipeline = pipeline
        self._allowed_dirs = allowed_dirs or ["data/uploads", "docs"]
        self._semantic_cache = semantic_cache
        self._object_store = object_store
        self._document_registry = document_registry
        self._task_store = task_store

    @property
    def name(self) -> str:
        return "document_ingest"

    @property
    def description(self) -> str:
        return "将 PDF 或 PPT 文件导入知识库，支持 .pdf 和 .pptx 格式"

    def get_args_schema(self) -> type[DocumentIngestArgs]:
        return DocumentIngestArgs

    def _get_pipeline(self, collection: str) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if self._settings is None:
            raise RuntimeError("DocumentIngestTool requires settings or a pre-built pipeline")
        from src.ingestion.pipeline import IngestionPipeline
        return IngestionPipeline(settings=self._settings, collection=collection)

    def _is_path_allowed(self, file_path: Path) -> bool:
        resolved = file_path.resolve()
        for d in self._allowed_dirs:
            base = Path(d).resolve()
            if str(resolved).startswith(str(base) + "/") or resolved == base:
                return True
        return False

    async def execute(self, context: ToolContext, args: DocumentIngestArgs) -> ToolResult:
        path = Path(args.file_path)
        if not path.exists():
            return ToolResult(success=False, error=f"文件不存在: {args.file_path}")

        if path.is_symlink():
            return ToolResult(success=False, error="不允许通过符号链接导入文件")

        if not self._is_path_allowed(path):
            return ToolResult(success=False, error="文件路径不在允许的目录范围内")

        ext = path.suffix.lower()
        if ext not in (".pdf", ".pptx"):
            return ToolResult(success=False, error=f"不支持的文件格式: {ext}，仅支持 .pdf 和 .pptx")

        uploaded_object_key = str(context.metadata.get("uploaded_object_key", "") or "")
        task_id = ""
        if self._task_store is not None:
            try:
                task_id = self._task_store.create_task(
                    file_path=str(path),
                    collection=args.collection,
                    object_uri=self._object_store.uri_for(uploaded_object_key) if uploaded_object_key and self._object_store else "",
                )
                self._task_store.update_status(task_id, "running")
            except Exception:
                logger.warning("Failed to create ingestion task record", exc_info=True)
                task_id = ""

        try:
            pipeline = self._get_pipeline(args.collection)
            result = await asyncio.to_thread(pipeline.run, str(path))

            if result.success:
                if self._semantic_cache is not None:
                    try:
                        self._semantic_cache.invalidate_by_collection(args.collection)
                    except Exception:
                        logger.debug("Semantic cache invalidation failed", exc_info=True)
                if self._document_registry is not None and result.doc_id:
                    try:
                        self._document_registry.upsert_document(
                            file_hash=result.doc_id,
                            file_path=str(path),
                            collection=args.collection,
                            status="ready",
                            object_uri=self._object_store.uri_for(uploaded_object_key) if uploaded_object_key and self._object_store else "",
                            metadata={
                                "chunk_count": result.chunk_count,
                                "image_count": result.image_count,
                                "task_id": task_id,
                            },
                        )
                    except Exception:
                        logger.warning("Failed to upsert document registry entry", exc_info=True)
                if task_id and self._task_store is not None:
                    try:
                        self._task_store.update_status(task_id, "succeeded")
                    except Exception:
                        logger.warning("Failed to update ingestion task status", exc_info=True)

                return ToolResult(
                    success=True,
                    result_for_llm=(
                        f"文件 '{path.name}' 入库成功！\n"
                        f"- 文档 ID: {result.doc_id}\n"
                        f"- 分块数: {result.chunk_count}\n"
                        f"- 图片数: {result.image_count}"
                    ),
                    metadata={
                        "doc_id": result.doc_id,
                        "chunk_count": result.chunk_count,
                        "image_count": result.image_count,
                        "task_id": task_id,
                        "object_uri": self._object_store.uri_for(uploaded_object_key) if uploaded_object_key and self._object_store else "",
                    },
                )
            else:
                if task_id and self._task_store is not None:
                    try:
                        self._task_store.update_status(task_id, "failed", result.error or "")
                    except Exception:
                        logger.warning("Failed to update ingestion task status", exc_info=True)
                return ToolResult(success=False, error=f"入库失败: {result.error}")

        except Exception as exc:
            logger.exception("DocumentIngestTool failed")
            if task_id and self._task_store is not None:
                try:
                    self._task_store.update_status(task_id, "failed", str(exc))
                except Exception:
                    logger.warning("Failed to update ingestion task status", exc_info=True)
            return ToolResult(success=False, error=f"入库异常: {exc}")
