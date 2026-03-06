"""Conflict detection data models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConflictType(str, Enum):
    FACTUAL = "factual"
    NUMERICAL = "numerical"
    DEFINITIONAL = "definitional"
    TEMPORAL = "temporal"


class Conflict(BaseModel):
    type: ConflictType
    chunk_a_id: str
    chunk_b_id: str
    claim_a: str
    claim_b: str
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""


class ConflictReport(BaseModel):
    conflicts: list[Conflict] = Field(default_factory=list)
    trusted_chunk_ids: list[str] = Field(default_factory=list)
    resolution_summary: str = ""

    @property
    def has_conflicts(self) -> bool:
        return len(self.conflicts) > 0
