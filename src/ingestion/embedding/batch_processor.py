"""Batch Processor for orchestrating dense and sparse encoding.

This module implements the Batch Processor component of the Ingestion Pipeline,
responsible for coordinating the encoding workflow and managing batch operations.

Design Principles:
- Orchestration: Coordinates DenseEncoder and SparseEncoder in unified workflow
- Config-Driven: Batch size from settings, not hardcoded
- Observable: Records batch timing and statistics via TraceContext
- Error Handling: Individual batch failures don't crash entire pipeline
- Deterministic: Same inputs produce same batching and results
- Decoupled: Dense and sparse encoding in independent try/except blocks
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

from src.core.types import Chunk
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch processing operation.

    Attributes:
        dense_vectors: List of dense embeddings (one per successfully encoded chunk)
        sparse_stats: List of term statistics (one per successfully encoded chunk)
        batch_count: Number of batches processed
        total_time: Total processing time in seconds
        successful_chunks: Number of successfully processed chunks
        failed_chunks: Number of chunks that failed processing
        successful_chunk_indices: Indices of chunks that were successfully encoded
    """
    dense_vectors: List[List[float]]
    sparse_stats: List[Dict[str, Any]]
    batch_count: int
    total_time: float
    successful_chunks: int
    failed_chunks: int
    successful_chunk_indices: List[int] = None

    def __post_init__(self):
        if self.successful_chunk_indices is None:
            self.successful_chunk_indices = list(range(self.successful_chunks))


class BatchProcessor:
    """Orchestrates batch processing of chunks through encoding pipeline.

    This processor manages the workflow of converting chunks into both dense
    and sparse representations. It divides chunks into batches, drives the
    encoders, and collects timing metrics.

    Design:
    - Stateless: No state maintained between process() calls
    - Decoupled Encodings: Dense and sparse encoding in independent try/except
    - Metrics Collection: Records batch-level timing for observability
    - Order Preservation: Output order matches input chunk order
    - Error Visibility: All errors logged with logger.error

    Example:
        >>> from src.libs.embedding.embedding_factory import EmbeddingFactory
        >>> from src.core.settings import load_settings
        >>>
        >>> settings = load_settings("config/settings.yaml")
        >>> embedding = EmbeddingFactory.create(settings)
        >>> dense_encoder = DenseEncoder(embedding, batch_size=2)
        >>> sparse_encoder = SparseEncoder()
        >>>
        >>> processor = BatchProcessor(
        ...     dense_encoder=dense_encoder,
        ...     sparse_encoder=sparse_encoder,
        ...     batch_size=2
        ... )
        >>>
        >>> chunks = [
        ...     Chunk(id="1", text="Hello", metadata={}),
        ...     Chunk(id="2", text="World", metadata={})
        ... ]
        >>> result = processor.process(chunks)
        >>> len(result.dense_vectors) == len(chunks)  # True
        >>> len(result.sparse_stats) == len(chunks)  # True
    """

    def __init__(
        self,
        dense_encoder: DenseEncoder,
        sparse_encoder: SparseEncoder,
        batch_size: int = 100,
    ):
        """Initialize BatchProcessor.

        Args:
            dense_encoder: DenseEncoder instance for embedding generation
            sparse_encoder: SparseEncoder instance for term statistics
            batch_size: Number of chunks to process per batch (default: 100)

        Raises:
            ValueError: If batch_size <= 0
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder
        self.batch_size = batch_size

    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> BatchResult:
        """Process chunks through dense and sparse encoding pipeline.

        Dense and sparse encoding are now in independent try/except blocks:
        - Dense failure does NOT skip sparse encoding for the same batch
        - Only chunks with successful dense encoding are included in results
        - Sparse failures for successfully dense-encoded chunks get empty stats

        Args:
            chunks: List of Chunk objects to process
            trace: Optional TraceContext for observability

        Returns:
            BatchResult containing vectors, statistics, and metrics

        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot process empty chunks list")

        start_time = time.time()

        # Create batches
        batches = self._create_batches(chunks)
        batch_count = len(batches)

        # Process all batches
        dense_vectors: List[List[float]] = []
        sparse_stats: List[Dict[str, Any]] = []
        successful_chunks = 0
        failed_chunks = 0
        successful_chunk_indices: List[int] = []
        global_offset = 0  # track position in original chunks list

        for batch_idx, batch in enumerate(batches):
            batch_start = time.time()
            batch_dense: List[List[float]] = []
            batch_sparse: List[Dict[str, Any]] = []
            dense_ok = False

            # --- Dense encoding (independent) ---
            try:
                batch_dense = self.dense_encoder.encode(batch, trace=trace)
                dense_ok = True
            except Exception as e:
                logger.error(
                    "Batch %d/%d dense encoding failed (%d chunks): %s",
                    batch_idx + 1, batch_count, len(batch), e,
                    exc_info=True,
                )
                if trace:
                    trace.record_stage(
                        f"batch_{batch_idx}_dense_error",
                        {"error": str(e), "batch_size": len(batch)}
                    )

            # --- Sparse encoding (independent) ---
            try:
                batch_sparse = self.sparse_encoder.encode(batch, trace=trace)
            except Exception as e:
                logger.error(
                    "Batch %d/%d sparse encoding failed (%d chunks): %s",
                    batch_idx + 1, batch_count, len(batch), e,
                    exc_info=True,
                )
                if trace:
                    trace.record_stage(
                        f"batch_{batch_idx}_sparse_error",
                        {"error": str(e), "batch_size": len(batch)}
                    )

            # --- Assemble results ---
            if dense_ok and len(batch_dense) == len(batch):
                dense_vectors.extend(batch_dense)
                # If sparse succeeded, use it; otherwise fill with empty stats
                if len(batch_sparse) == len(batch):
                    sparse_stats.extend(batch_sparse)
                else:
                    for c in batch:
                        sparse_stats.append({
                            "chunk_id": c.id,
                            "term_frequencies": {},
                            "doc_length": 0,
                            "unique_terms": 0,
                        })
                successful_chunks += len(batch)
                for i in range(len(batch)):
                    successful_chunk_indices.append(global_offset + i)
            else:
                failed_chunks += len(batch)

            batch_duration = time.time() - batch_start

            if trace:
                trace.record_stage(
                    f"batch_{batch_idx}",
                    {
                        "batch_size": len(batch),
                        "duration_seconds": batch_duration,
                        "chunks_processed": len(batch),
                        "dense_ok": dense_ok,
                    }
                )

            global_offset += len(batch)

        total_time = time.time() - start_time

        # Record overall processing statistics
        if trace:
            trace.record_stage(
                "batch_processing",
                {
                    "total_chunks": len(chunks),
                    "batch_count": batch_count,
                    "batch_size": self.batch_size,
                    "successful_chunks": successful_chunks,
                    "failed_chunks": failed_chunks,
                    "total_time_seconds": total_time
                }
            )

        if failed_chunks > 0:
            logger.warning(
                "Batch processing completed with %d/%d chunks failed",
                failed_chunks, len(chunks),
            )

        return BatchResult(
            dense_vectors=dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=batch_count,
            total_time=total_time,
            successful_chunks=successful_chunks,
            failed_chunks=failed_chunks,
            successful_chunk_indices=successful_chunk_indices,
        )

    def _create_batches(self, chunks: List[Chunk]) -> List[List[Chunk]]:
        """Divide chunks into batches of specified size.

        Args:
            chunks: List of chunks to batch

        Returns:
            List of batches, where each batch is a list of chunks.
            Order is preserved.
        """
        batches = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def get_batch_count(self, total_chunks: int) -> int:
        """Calculate number of batches for given chunk count."""
        if total_chunks <= 0:
            return 0
        return (total_chunks + self.batch_size - 1) // self.batch_size
