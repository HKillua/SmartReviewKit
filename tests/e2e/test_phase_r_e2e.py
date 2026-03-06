"""Phase R: Ingestion Pipeline 深度优化 — 端到端测试

验证各优化在集成场景下的正确行为。
"""

from __future__ import annotations

import gzip
import json
import tempfile
import time
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.types import Chunk

_DEFAULT_META = {"source_path": "test.pdf", "chunk_index": 0}


def _make_chunk(id: str, text: str, **extra_meta) -> Chunk:
    meta = {**_DEFAULT_META, **extra_meta}
    return Chunk(id=id, text=text, metadata=meta)


class TestE2EBMM25FullCycle(TestCase):
    """E2E: Build -> query -> add_document -> remove -> query."""

    def test_full_lifecycle(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        tmp = tempfile.mkdtemp()
        idx = BM25Indexer(index_dir=tmp)

        batch1 = [
            {"chunk_id": "doc1_0", "term_frequencies": {"tcp": 3, "protocol": 1}, "doc_length": 4},
            {"chunk_id": "doc1_1", "term_frequencies": {"tcp": 1, "handshake": 2}, "doc_length": 3},
        ]
        idx.build(batch1, collection="net")
        self.assertEqual(idx._metadata["num_docs"], 2)

        results = idx.query(["tcp"], top_k=5)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "doc1_0")

        batch2 = [
            {"chunk_id": "doc2_0", "term_frequencies": {"udp": 2, "protocol": 1}, "doc_length": 3},
        ]
        idx.add_document(batch2, collection="net")
        self.assertEqual(idx._metadata["num_docs"], 3)

        results = idx.query(["udp"], top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "doc2_0")

        idx2 = BM25Indexer(index_dir=tmp)
        self.assertTrue(idx2.load(collection="net"))
        self.assertEqual(idx2._metadata["num_docs"], 3)

        idx2.remove_document("doc1_", collection="net")
        self.assertEqual(idx2._metadata["num_docs"], 1)

        results = idx2.query(["tcp"], top_k=5)
        self.assertEqual(len(results), 0)
        results = idx2.query(["udp"], top_k=5)
        self.assertEqual(len(results), 1)


class TestE2EChunkerParentChild(TestCase):
    """E2E: Parent-child chunking produces collision-free IDs."""

    def test_large_document_parent_child_ids_unique(self):
        from src.ingestion.chunking.document_chunker import DocumentChunker
        from src.core.types import Document

        chunker = DocumentChunker.__new__(DocumentChunker)
        chunker._parent_child = True
        chunker._parent_size = 200
        chunker._parent_overlap = 20
        splitter = MagicMock()
        splitter.split_text.return_value = [f"word_{i} " * 10 for i in range(50)]
        chunker._splitter = splitter

        doc = Document(
            id="bigdoc",
            text="test content " * 500,
            metadata={"source_path": "test.pdf"},
        )
        chunks = chunker.split_document(doc)
        ids = [c.id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)), f"Duplicate IDs found among {len(ids)} chunks")

        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if not c.metadata.get("is_parent", True)]
        self.assertGreater(len(parents), 0)
        self.assertGreater(len(children), 0)

        for child in children:
            self.assertIsNotNone(child.parent_id)


class TestE2ESimHashPerformance(TestCase):
    """E2E: SimHash dedup handles large chunk sets efficiently."""

    def test_1000_chunks_under_5_seconds(self):
        from src.ingestion.transform.chunk_dedup import dedup_chunks

        chunks = [
            _make_chunk(str(i), f"unique content about topic number {i} " * 20, chunk_index=i)
            for i in range(1000)
        ]
        for i in range(50):
            chunks.append(_make_chunk(
                f"dup_{i}",
                f"unique content about topic number {i} " * 20,
                chunk_index=1000 + i,
            ))

        start = time.monotonic()
        result = dedup_chunks(chunks, threshold=3, num_bands=4)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 5.0, f"Dedup took {elapsed:.2f}s")
        self.assertLess(len(result), len(chunks))


class TestE2EDenseEncoderRetryIntegration(TestCase):
    """E2E: DenseEncoder retries through BatchProcessor."""

    def test_batch_processor_with_retrying_encoder(self):
        from src.ingestion.embedding.batch_processor import BatchProcessor
        from src.ingestion.embedding.dense_encoder import DenseEncoder
        from src.ingestion.embedding.sparse_encoder import SparseEncoder

        call_count = 0

        def mock_embed(texts, trace=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient network error")
            return [[0.1] * 3 for _ in texts]

        mock_embedding = MagicMock()
        mock_embedding.embed.side_effect = mock_embed

        dense = DenseEncoder(mock_embedding, batch_size=10, max_retries=2)
        sparse = SparseEncoder()
        bp = BatchProcessor(dense_encoder=dense, sparse_encoder=sparse, batch_size=10)

        chunks = [
            _make_chunk("c1", "TCP is a transport layer protocol", chunk_index=0),
            _make_chunk("c2", "UDP is connectionless", chunk_index=1),
        ]
        result = bp.process(chunks)

        self.assertEqual(result.successful_chunks, 2)
        self.assertEqual(len(result.dense_vectors), 2)
        self.assertEqual(len(result.sparse_stats), 2)


class TestE2EVectorUpserterIDConsistency(TestCase):
    """E2E: Chunk IDs from DocumentChunker flow through to VectorUpserter."""

    def test_ids_match_end_to_end(self):
        from src.ingestion.chunking.document_chunker import DocumentChunker
        from src.ingestion.storage.vector_upserter import VectorUpserter
        from src.core.types import Document

        chunker = DocumentChunker.__new__(DocumentChunker)
        chunker._parent_child = False
        splitter = MagicMock()
        splitter.split_text.return_value = ["chunk one text", "chunk two text"]
        chunker._splitter = splitter

        doc = Document(id="testdoc", text="chunk one text\nchunk two text", metadata={"source_path": "test.pdf"})
        chunks = chunker.split_document(doc)

        mock_store = MagicMock()
        with patch("src.ingestion.storage.vector_upserter.VectorStoreFactory") as mock_factory:
            mock_factory.create.return_value = mock_store
            upserter = VectorUpserter(MagicMock())

        vectors = [[0.1] * 3, [0.2] * 3]
        ids = upserter.upsert(chunks, vectors)

        self.assertEqual(ids[0], chunks[0].id)
        self.assertEqual(ids[1], chunks[1].id)
        records = mock_store.upsert.call_args[0][0]
        self.assertEqual(records[0]["id"], chunks[0].id)
        self.assertEqual(records[1]["id"], chunks[1].id)
