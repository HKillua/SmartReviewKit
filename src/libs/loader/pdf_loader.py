"""PDF Loader with math post-processing, chapter detection, and exercise recognition."""

from __future__ import annotations

import hashlib
import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

from PIL import Image
import io

from src.core.types import Document, DocumentSection
from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.math_utils import postprocess_math

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_CN_CHAPTER_RE = re.compile(
    r"^第[一二三四五六七八九十百\d]+[章节][\s:：]*(.*)", re.MULTILINE
)
_NUM_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.*)", re.MULTILINE)
_EXERCISE_KW_RE = re.compile(r"习题|练习|思考题|Exercise|Problem|Questions", re.IGNORECASE)
_ANSWER_KW_RE = re.compile(r"答案|解答|Solution|Answer", re.IGNORECASE)
_DEFINITION_RE = re.compile(r"定义|Definition", re.IGNORECASE)
_THEOREM_RE = re.compile(r"定理|Theorem|引理|Lemma", re.IGNORECASE)
_FORMULA_DELIM_RE = re.compile(r"\$\$?.+?\$\$?", re.DOTALL)


def _infer_content_type(text: str) -> str:
    preview = text[:300]
    if _EXERCISE_KW_RE.search(preview):
        return "exercise"
    if _ANSWER_KW_RE.search(preview):
        return "exercise"
    if _DEFINITION_RE.search(preview):
        return "definition"
    if _THEOREM_RE.search(preview):
        return "theorem"
    return "concept"


class PdfLoader(BaseLoader):
    """PDF Loader using MarkItDown with OCR fallback for scanned textbooks."""

    def __init__(
        self,
        extract_images: bool = True,
        image_storage_dir: str | Path = "data/images",
    ):
        if not MARKITDOWN_AVAILABLE:
            raise ImportError("MarkItDown is required for PdfLoader. Install with: pip install markitdown")
        self.extract_images = extract_images
        self.image_storage_dir = Path(image_storage_dir)
        self._markitdown = MarkItDown()

    def load(self, file_path: str | Path) -> Document:
        path = self._validate_file(file_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {path}")

        doc_hash = self._compute_file_hash(path)
        doc_id = f"doc_{doc_hash[:16]}"

        try:
            result = self._markitdown.convert(str(path))
            text_content = result.text_content if hasattr(result, "text_content") else str(result)
        except Exception as e:
            logger.error(f"Failed to parse PDF {path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e

        used_ocr_fallback = False

        # MarkItDown occasionally returns near-empty output on textbook PDFs.
        # When that happens and PyMuPDF is available, fall back to direct page
        # text extraction so production ingestion does not fail with zero-text
        # documents.
        if not text_content or not text_content.strip():
            fallback_text = self._extract_text_with_pymupdf(path)
            if fallback_text:
                logger.info("PDF text fallback via PyMuPDF succeeded for %s", path)
                text_content = fallback_text
            else:
                ocr_text = self._extract_text_with_rapidocr(path)
                if ocr_text:
                    logger.info("PDF text fallback via RapidOCR succeeded for %s", path)
                    text_content = ocr_text
                    used_ocr_fallback = True

        # Math post-processing
        text_content = postprocess_math(text_content)

        metadata: Dict[str, Any] = {
            "source_path": str(path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "ocr_fallback_used": used_ocr_fallback,
        }

        title = self._extract_title(text_content)
        if title:
            metadata["title"] = title

        # Image extraction
        if self.extract_images and not used_ocr_fallback:
            try:
                text_content, images_metadata = self._extract_and_process_images(
                    path, text_content, doc_hash
                )
                metadata["images"] = images_metadata
            except Exception as e:
                logger.warning(f"Image extraction failed for {path}, continuing text-only: {e}")
                metadata["images"] = []
        elif used_ocr_fallback:
            # Scanned PDFs are page-image based; extracting those images would
            # only trigger redundant captioning work on the page backgrounds.
            metadata["images"] = []

        # Structure detection → sections
        sections = self._detect_sections(text_content)
        if sections:
            metadata["sections"] = [s.to_dict() for s in sections]

        return Document(id=doc_id, text=text_content, metadata=metadata)

    # ------------------------------------------------------------------
    # Section / chapter detection
    # ------------------------------------------------------------------

    def _detect_sections(self, text: str) -> list[DocumentSection]:
        boundaries: list[tuple[int, int, str, int]] = []  # (pos, level, title, line_idx)

        for m in _HEADING_RE.finditer(text):
            level = len(m.group(1))
            boundaries.append((m.start(), level, m.group(2).strip(), 0))

        for m in _CN_CHAPTER_RE.finditer(text):
            boundaries.append((m.start(), 1, m.group(0).strip(), 0))

        for m in _NUM_SECTION_RE.finditer(text):
            depth = m.group(1).count(".")
            boundaries.append((m.start(), min(depth + 1, 3), m.group(0).strip(), 0))

        if not boundaries:
            return []

        boundaries.sort(key=lambda x: x[0])
        # Deduplicate overlapping matches at same position
        seen_positions: set[int] = set()
        unique: list[tuple[int, int, str, int]] = []
        for b in boundaries:
            if b[0] not in seen_positions:
                seen_positions.add(b[0])
                unique.append(b)
        boundaries = unique

        sections: list[DocumentSection] = []
        for i, (pos, level, title, _) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            content = text[pos:end].strip()
            has_formula = bool(_FORMULA_DELIM_RE.search(content))
            ct = _infer_content_type(title + " " + content[:300])
            if has_formula and ct == "concept" and len(content) < 400:
                ct = "formula"

            sections.append(DocumentSection(
                title=title,
                level=level,
                content=content,
                content_type=ct,
                page_or_slide=0,
                has_formula=has_formula,
            ))

        return sections

    # ------------------------------------------------------------------
    # Helpers (unchanged logic, cleaned up)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for line in text.split("\n")[:20]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        for line in text.split("\n")[:10]:
            line = line.strip()
            if line:
                return line
        return None

    @staticmethod
    def _generate_image_id(doc_hash: str, page_num: int, img_index: int) -> str:
        return f"{doc_hash[:8]}_{page_num}_{img_index}"

    def _extract_and_process_images(
        self, pdf_path: Path, text_content: str, doc_hash: str,
    ) -> tuple[str, List[Dict[str, Any]]]:
        if not PYMUPDF_AVAILABLE:
            logger.warning(f"PyMuPDF not available, skipping image extraction for {pdf_path}")
            return text_content, []

        images_metadata: list[Dict[str, Any]] = []
        modified_text = text_content

        try:
            image_dir = self.image_storage_dir / doc_hash
            image_dir.mkdir(parents=True, exist_ok=True)

            # Split text by page markers so we can insert placeholders near
            # the pages where the images actually appear.
            page_marker_re = re.compile(r"(\n---\s*Page\s+\d+\s*---\n)", re.IGNORECASE)
            text_segments = page_marker_re.split(modified_text)

            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_placeholders: list[str] = []
                    for img_index, img_info in enumerate(page.get_images(full=True)):
                        try:
                            xref = img_info[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]

                            image_id = self._generate_image_id(doc_hash, page_num + 1, img_index + 1)
                            image_path = image_dir / f"{image_id}.{image_ext}"
                            image_path.write_bytes(image_bytes)

                            placeholder = f"[IMAGE: {image_id}]"
                            page_placeholders.append(placeholder)

                            try:
                                img = Image.open(io.BytesIO(image_bytes))
                                width, height = img.size
                            except Exception:
                                width, height = 0, 0

                            try:
                                relative_path = image_path.relative_to(Path.cwd())
                            except ValueError:
                                relative_path = image_path.absolute()

                            images_metadata.append({
                                "id": image_id,
                                "path": str(relative_path),
                                "page": page_num + 1,
                                "text_offset": 0,
                                "text_length": len(placeholder),
                                "position": {
                                    "width": width, "height": height,
                                    "page": page_num + 1, "index": img_index,
                                },
                            })
                        except Exception as e:
                            logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")

                    # Append page-level placeholders to end of text
                    if page_placeholders:
                        suffix = "\n" + "\n".join(page_placeholders) + "\n"
                        modified_text += suffix

        except Exception as e:
            logger.warning(f"Image extraction failed for {pdf_path}: {e}")
            return text_content, []

        if images_metadata:
            logger.info(f"Extracted {len(images_metadata)} images from {pdf_path}")
        return modified_text, images_metadata

    def _extract_text_with_pymupdf(self, pdf_path: Path) -> str:
        if not PYMUPDF_AVAILABLE:
            return ""
        pages: list[str] = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    page_text = (page.get_text("text") or "").strip()
                    if not page_text:
                        continue
                    pages.append(f"\n--- Page {page_num} ---\n{page_text}\n")
        except Exception as exc:
            logger.warning("PyMuPDF text fallback failed for %s: %s", pdf_path, exc)
            return ""
        return "\n".join(pages).strip()

    @cached_property
    def _rapidocr(self) -> Any | None:
        if not RAPIDOCR_AVAILABLE:
            return None
        try:
            return RapidOCR()
        except Exception as exc:
            logger.warning("RapidOCR initialization failed: %s", exc)
            return None

    def _extract_text_with_rapidocr(self, pdf_path: Path) -> str:
        if not PYMUPDF_AVAILABLE:
            return ""
        ocr_engine = self._rapidocr
        if ocr_engine is None:
            return ""

        pages: list[str] = []
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                for page_num, page in enumerate(doc, start=1):
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image_bytes = pix.tobytes("png")
                    try:
                        result, _ = ocr_engine(image_bytes)
                    except Exception as exc:
                        logger.warning("RapidOCR failed on %s page %d: %s", pdf_path.name, page_num, exc)
                        continue

                    lines = [
                        str(item[1]).strip()
                        for item in (result or [])
                        if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[1]).strip()
                    ]
                    if lines:
                        pages.append(f"\n--- Page {page_num} ---\n" + "\n".join(lines) + "\n")
                    if page_num % 20 == 0 or page_num == total_pages:
                        logger.info(
                            "RapidOCR progress for %s: %d/%d pages",
                            pdf_path.name,
                            page_num,
                            total_pages,
                        )
        except Exception as exc:
            logger.warning("RapidOCR text fallback failed for %s: %s", pdf_path, exc)
            return ""
        return "\n".join(pages).strip()
