"""Document chunking module with optional Parent-Child hierarchy.

Transforms Document objects into Chunk objects with proper ID generation,
metadata inheritance, and traceability.  When ``parent_child_enabled`` is
set in ingestion config, each document is first split into coarse *parent*
chunks, then each parent is further split into finer *child* chunks whose
``parent_id`` points back to the parent.  This enables parent-retrieval
strategies at query time.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, List, Optional

from src.core.types import Chunk, Document
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """Converts Documents into Chunks with optional parent-child hierarchy."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._splitter = SplitterFactory.create(settings)

        self._parent_child = False
        self._parent_size = 2400
        self._parent_overlap = 200
        if hasattr(settings, "ingestion") and settings.ingestion:
            raw = getattr(settings.ingestion, "chunk_refiner", None) or {}
            self._parent_child = bool(raw.get("parent_child_enabled", False))
            self._parent_size = int(raw.get("parent_chunk_size", 2400))
            self._parent_overlap = int(raw.get("parent_chunk_overlap", 200))

    def split_document(self, document: Document) -> List[Chunk]:
        if not document.text or not document.text.strip():
            raise ValueError(f"Document {document.id} has no text content to split")

        if self._parent_child:
            return self._split_with_parent_child(document)

        return self._split_flat(document)

    # ------------------------------------------------------------------
    # Flat splitting (original behaviour)
    # ------------------------------------------------------------------

    def _split_flat(self, document: Document) -> List[Chunk]:
        text_fragments = self._splitter.split_text(document.text)
        if not text_fragments:
            raise ValueError(f"Splitter returned no chunks for document {document.id}")

        chunks: List[Chunk] = []
        for index, text in enumerate(text_fragments):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            chunk_metadata = self._inherit_metadata(document, index, text)
            chunks.append(Chunk(id=chunk_id, text=text, metadata=chunk_metadata))
        return chunks

    # ------------------------------------------------------------------
    # Parent-child splitting
    # ------------------------------------------------------------------

    def _split_with_parent_child(self, document: Document) -> List[Chunk]:
        parent_fragments = self._coarse_split(document.text)
        all_chunks: List[Chunk] = []

        for p_idx, parent_text in enumerate(parent_fragments):
            parent_id = self._generate_chunk_id(document.id, p_idx, parent_text, prefix="parent")
            parent_meta = self._inherit_metadata(document, p_idx, parent_text)
            parent_meta["is_parent"] = True
            parent_chunk = Chunk(
                id=parent_id, text=parent_text, metadata=parent_meta,
            )
            all_chunks.append(parent_chunk)

            child_fragments = self._splitter.split_text(parent_text)
            for c_idx, child_text in enumerate(child_fragments):
                child_id = self._generate_chunk_id(
                    document.id, p_idx * 1000 + c_idx, child_text, prefix="child"
                )
                child_meta = self._inherit_metadata(document, p_idx * 1000 + c_idx, child_text)
                child_meta["is_parent"] = False
                child_meta["parent_chunk_id"] = parent_id
                child_chunk = Chunk(
                    id=child_id, text=child_text, metadata=child_meta,
                    parent_id=parent_id,
                )
                all_chunks.append(child_chunk)

        return all_chunks

    def _coarse_split(self, text: str) -> List[str]:
        """Split text into large parent-level segments."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._parent_size,
                chunk_overlap=self._parent_overlap,
                separators=["\n\n---\n\n", "\n\n", "\n", ". ", " "],
            )
            return splitter.split_text(text)
        except ImportError:
            step = self._parent_size - self._parent_overlap
            return [text[i: i + self._parent_size] for i in range(0, len(text), step)]

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _generate_chunk_id(
        self, doc_id: str, index: int, text: str, prefix: str = ""
    ) -> str:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        tag = f"{prefix}_" if prefix else ""
        return f"{doc_id}_{tag}{index:04d}_{content_hash}"

    def _inherit_metadata(self, document: Document, chunk_index: int, chunk_text: str = "") -> dict:
        chunk_metadata = document.metadata.copy()
        doc_images = document.metadata.get("images", [])
        chunk_metadata.pop("images", None)
        chunk_metadata.pop("sections", None)

        chunk_metadata["chunk_index"] = chunk_index
        chunk_metadata["source_ref"] = document.id

        image_refs = []
        if chunk_text:
            pattern = r'\[IMAGE:\s*([^\]]+)\]'
            matches = re.findall(pattern, chunk_text)
            image_refs = [m.strip() for m in matches]
        chunk_metadata["image_refs"] = image_refs

        chunk_images = []
        if image_refs and doc_images:
            image_lookup = {img.get("id"): img for img in doc_images}
            for img_id in image_refs:
                if img_id in image_lookup:
                    chunk_images.append(image_lookup[img_id])
        if chunk_images:
            chunk_metadata["images"] = chunk_images
            chunk_metadata["page_num"] = chunk_images[0].get("page")

        return chunk_metadata
