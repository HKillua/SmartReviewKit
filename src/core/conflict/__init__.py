"""Agent-RAG Conflict Detection — identifies factual contradictions across retrieved chunks."""

from src.core.conflict.types import Conflict, ConflictReport, ConflictType
from src.core.conflict.detector import ConflictDetector

__all__ = ["Conflict", "ConflictReport", "ConflictType", "ConflictDetector"]
