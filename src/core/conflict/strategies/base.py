"""Abstract base for conflict detection strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.core.conflict.types import Conflict
from src.core.types import RetrievalResult


class ConflictStrategy(ABC):
    """Each strategy inspects a list of retrieval results and returns conflicts."""

    @abstractmethod
    async def detect(self, query: str, results: List[RetrievalResult]) -> List[Conflict]:
        ...
