"""
Response Module.

This package contains response building components:
- Response builder
- Citation generator
- Multimodal assembler
"""

from src.core.response.citation_generator import Citation, CitationGenerator

try:
    from src.core.response.multimodal_assembler import (
        ImageContent,
        ImageReference,
        MultimodalAssembler,
    )
    from src.core.response.response_builder import MCPToolResponse, ResponseBuilder
except ModuleNotFoundError:  # pragma: no cover - optional MCP dependency
    ImageContent = None
    ImageReference = None
    MultimodalAssembler = None
    MCPToolResponse = None
    ResponseBuilder = None

__all__ = [
    "Citation",
    "CitationGenerator",
    "ImageContent",
    "ImageReference",
    "MCPToolResponse",
    "MultimodalAssembler",
    "ResponseBuilder",
]
