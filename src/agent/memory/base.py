"""Memory system base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.agent.types import Message


class ConversationFilter(ABC):
    """Filters / compresses conversation history before sending to LLM."""

    @abstractmethod
    def filter_messages(self, messages: list[Message]) -> list[Message]:
        ...
