"""PPTX (PowerPoint) document loader — extracts text, tables, and speaker notes."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader

logger = logging.getLogger(__name__)


class PptxLoader(BaseLoader):
    """Load .pptx files into a Document with Markdown-formatted content."""

    def __init__(self, extract_images: bool = False, image_storage_dir: str = "data/images") -> None:
        self._extract_images = extract_images
        self._image_dir = Path(image_storage_dir)

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)

        try:
            from pptx import Presentation
        except ImportError as exc:
            raise ImportError("python-pptx is required: pip install python-pptx") from exc

        prs = Presentation(str(path))

        doc_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        parts: list[str] = []
        images: list[dict[str, Any]] = []
        image_seq = 0

        for slide_idx, slide in enumerate(prs.slides, 1):
            slide_title = ""
            slide_texts: list[str] = []

            for shape in slide.shapes:
                if shape.has_text_frame:
                    text = shape.text_frame.text.strip()
                    if not text:
                        continue
                    if shape.shape_id == slide.shapes.title.shape_id if slide.shapes.title else False:
                        slide_title = text
                    else:
                        slide_texts.append(text)

                if shape.has_table:
                    slide_texts.append(self._table_to_markdown(shape.table))

                if self._extract_images and shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE
                    image_seq += 1
                    img_id = f"{doc_hash}_{slide_idx}_{image_seq}"
                    placeholder = f"[IMAGE: {img_id}]"
                    slide_texts.append(placeholder)

                    img_path = self._save_image(shape, img_id)
                    images.append({"id": img_id, "page": slide_idx, "path": img_path})

            # Speaker notes
            notes_text = ""
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_text = slide.notes_slide.notes_text_frame.text.strip()

            # Assemble slide section
            header = f"## Slide {slide_idx}"
            if slide_title:
                header += f": {slide_title}"
            section = [header, ""]
            section.extend(slide_texts)
            if notes_text:
                section.append(f"\n> **备注**: {notes_text}")
            parts.append("\n".join(section))

        full_text = f"# {path.stem}\n\n" + "\n\n---\n\n".join(parts)

        return Document(
            id=doc_hash,
            text=full_text,
            metadata={
                "source_path": str(path),
                "doc_type": "pptx",
                "title": path.stem,
                "slide_count": len(prs.slides),
                "images": images,
            },
        )

    def _save_image(self, shape: Any, img_id: str) -> str:
        """Extract image bytes from a Picture shape and save to disk."""
        self._image_dir.mkdir(parents=True, exist_ok=True)
        blob = shape.image.blob
        ext = shape.image.content_type.split("/")[-1] if shape.image.content_type else "png"
        if ext == "jpeg":
            ext = "jpg"
        out_path = self._image_dir / f"{img_id}.{ext}"
        out_path.write_bytes(blob)
        return str(out_path)

    @staticmethod
    def _table_to_markdown(table: Any) -> str:
        rows: list[list[str]] = []
        for row in table.rows:
            rows.append([cell.text.strip() for cell in row.cells])
        if not rows:
            return ""
        lines = ["| " + " | ".join(rows[0]) + " |"]
        lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
