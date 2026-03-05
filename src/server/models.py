"""FastAPI request / response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: Optional[str] = None
    user_id: str = Field(default="default_user")


class ChatStreamChunk(BaseModel):
    type: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class UploadResponse(BaseModel):
    success: bool
    filename: str = ""
    doc_id: str = ""
    chunk_count: int = 0
    image_count: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    skills_loaded: int = 0
    tools_registered: int = 0
