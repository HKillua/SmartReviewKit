#!/usr/bin/env python3
"""Batch ingest PPTX files from docs/computer_internet/ into the knowledge base."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionPipeline, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest_pptx")

PPTX_DIR = Path(__file__).resolve().parents[1] / "docs" / "computer_internet"
COLLECTION = "computer_network"


def main() -> None:
    pptx_files = sorted(PPTX_DIR.glob("*.pptx"))
    if not pptx_files:
        logger.error("No .pptx files found in %s", PPTX_DIR)
        sys.exit(1)

    logger.info("Found %d PPTX files to ingest", len(pptx_files))
    for f in pptx_files:
        logger.info("  - %s (%.1f MB)", f.name, f.stat().st_size / 1024 / 1024)

    settings = load_settings()
    pipeline = IngestionPipeline(settings, collection=COLLECTION, force=False)

    results: list[PipelineResult] = []
    t0 = time.monotonic()

    for idx, pptx_file in enumerate(pptx_files, 1):
        logger.info("\n{'='*60}")
        logger.info("[%d/%d] Ingesting: %s", idx, len(pptx_files), pptx_file.name)
        result = pipeline.run(str(pptx_file))
        results.append(result)

        status = "OK" if result.success else f"FAIL: {result.error}"
        logger.info("[%d/%d] %s — chunks=%d, %s", idx, len(pptx_files), pptx_file.name, result.chunk_count, status)

    elapsed = time.monotonic() - t0
    pipeline.close()

    logger.info("\n" + "=" * 60)
    logger.info("Batch ingestion complete in %.1fs", elapsed)
    success = sum(1 for r in results if r.success)
    total_chunks = sum(r.chunk_count for r in results)
    logger.info("  Success: %d/%d files", success, len(results))
    logger.info("  Total chunks: %d", total_chunks)

    for r in results:
        icon = "✅" if r.success else "❌"
        logger.info("  %s %s — chunks=%d", icon, Path(r.file_path).name, r.chunk_count)

    if success < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
