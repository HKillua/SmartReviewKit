"""Splitter module - text splitting strategies with pluggable providers."""

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import SplitterFactory

try:
    from src.libs.splitter.recursive_splitter import RecursiveSplitter
except ImportError:
    RecursiveSplitter = None  # type: ignore[assignment,misc]

try:
    from src.libs.splitter.semantic_splitter import SemanticSplitter
except ImportError:
    SemanticSplitter = None  # type: ignore[assignment,misc]

try:
    from src.libs.splitter.structure_splitter import StructureAwareSplitter
except ImportError:
    StructureAwareSplitter = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseSplitter",
    "SplitterFactory",
    "RecursiveSplitter",
    "SemanticSplitter",
    "StructureAwareSplitter",
]
