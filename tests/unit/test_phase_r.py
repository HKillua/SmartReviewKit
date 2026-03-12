"""Phase R: Ingestion Pipeline 深度优化 — 单元测试

覆盖 R1-R19 的所有修复项。
"""

from __future__ import annotations

import gzip
import json
import math
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest import TestCase
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.core.types import Chunk

_DEFAULT_META = {"source_path": "test.pdf", "chunk_index": 0}


def _make_chunk(id: str, text: str, **extra_meta) -> Chunk:
    meta = {**_DEFAULT_META, **extra_meta}
    return Chunk(id=id, text=text, metadata=meta)


# ====================================================================
# R1: BM25 IDF — Lucene-style formula + corrupted-stats guard
# ====================================================================

class TestR1BM25IDF(TestCase):

    def _make_indexer(self, **kw):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        return BM25Indexer(index_dir=tempfile.mkdtemp(), **kw)

    def test_idf_all_docs_contain_term(self):
        idx = self._make_indexer()
        result = idx._calculate_idf(num_docs=10, df=10)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)

    def test_idf_df_exceeds_num_docs(self):
        """Invalid df > N should not crash even though the ratio is non-positive."""
        idx = self._make_indexer()
        result = idx._calculate_idf(num_docs=5, df=8)
        self.assertIsInstance(result, float)
        self.assertFalse(math.isnan(result))

    def test_idf_rare_term(self):
        idx = self._make_indexer()
        rare = idx._calculate_idf(num_docs=100, df=1)
        common = idx._calculate_idf(num_docs=100, df=50)
        self.assertGreater(rare, common)

    def test_idf_zero_docs(self):
        idx = self._make_indexer()
        result = idx._calculate_idf(num_docs=0, df=0)
        self.assertIsInstance(result, float)

    def test_idf_single_doc_no_crash(self):
        idx = self._make_indexer()
        result = idx._calculate_idf(num_docs=1, df=1)
        self.assertIsInstance(result, float)
        self.assertFalse(math.isnan(result))

    def test_idf_formula_is_lucene_variant(self):
        """Verify Lucene-style BM25 log(1 + ...) formula remains unchanged."""
        idx = self._make_indexer()
        result = idx._calculate_idf(num_docs=10, df=5)
        expected = math.log(1 + ((10 - 5 + 0.5) / (5 + 0.5)))
        self.assertAlmostEqual(result, expected)


# ====================================================================
# R2: remove_document total_length dedup
# ====================================================================

class TestR2TotalLengthDedup(TestCase):

    def test_remove_document_correct_avg_length(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        idx = BM25Indexer(index_dir=tempfile.mkdtemp())
        stats = [
            {"chunk_id": "a_0", "term_frequencies": {"tcp": 2, "ip": 1}, "doc_length": 3},
            {"chunk_id": "b_0", "term_frequencies": {"tcp": 1, "udp": 1}, "doc_length": 2},
        ]
        idx.build(stats, collection="test")
        idx.remove_document("a_", collection="test")
        self.assertEqual(idx._metadata["num_docs"], 1)
        self.assertAlmostEqual(idx._metadata["avg_doc_length"], 2.0)

    def test_remove_no_double_count(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        idx = BM25Indexer(index_dir=tempfile.mkdtemp())
        stats = [
            {"chunk_id": "x_0", "term_frequencies": {"a": 1, "b": 1, "c": 1}, "doc_length": 10},
            {"chunk_id": "y_0", "term_frequencies": {"a": 1}, "doc_length": 5},
        ]
        idx.build(stats, collection="test")
        idx.remove_document("y_", collection="test")
        self.assertAlmostEqual(idx._metadata["avg_doc_length"], 10.0)


# ====================================================================
# R3: Pipeline except NameError guard
# ====================================================================

class TestR3PipelineExceptGuard(TestCase):

    @patch("src.ingestion.pipeline.ImageStorage")
    @patch("src.ingestion.pipeline.create_sparse_index")
    @patch("src.ingestion.pipeline.VectorUpserter")
    @patch("src.ingestion.pipeline.BatchProcessor")
    @patch("src.ingestion.pipeline.ImageCaptioner")
    @patch("src.ingestion.pipeline.MetadataEnricher")
    @patch("src.ingestion.pipeline.ChunkRefiner")
    @patch("src.ingestion.pipeline.DocumentChunker")
    @patch("src.ingestion.pipeline.PdfLoader")
    @patch("src.ingestion.pipeline.PptxLoader")
    @patch("src.ingestion.pipeline.create_ingestion_backends")
    @patch("src.ingestion.pipeline.EmbeddingFactory")
    @patch("src.ingestion.pipeline.VectorStoreFactory")
    @patch("src.ingestion.pipeline.DenseEncoder")
    @patch("src.ingestion.pipeline.SparseEncoder")
    @patch("src.ingestion.pipeline.ContextualEnricher")
    def test_early_failure_before_hash(self, *mocks):
        from src.ingestion.pipeline import IngestionPipeline

        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.batch_size = 10
        settings.ingestion.chunk_refiner = {}
        settings.embedding = MagicMock()
        settings.embedding.provider = "mock"
        settings.vector_store = MagicMock()
        settings.vector_store.provider = "mock"
        settings.retrieval = MagicMock()
        settings.retrieval.contextual_enrichment = "rule"
        settings.retrieval.dedup_enabled = False

        backends = MagicMock()
        backends.object_store = MagicMock()
        backends.integrity_checker = MagicMock()
        backends.document_registry = MagicMock()
        backends.task_store = MagicMock()
        backends.image_storage = MagicMock()
        mocks[5].return_value = backends

        pipeline = IngestionPipeline(settings, collection="test")
        pipeline.integrity_checker.compute_sha256.side_effect = FileNotFoundError("no file")
        result = pipeline.run("/nonexistent/file.pdf")
        self.assertFalse(result.success)
        self.assertIsNone(result.doc_id)


# ====================================================================
# R4: Unified Chunk ID — VectorUpserter uses chunk.id
# ====================================================================

class TestR4UnifiedChunkID(TestCase):

    def test_upsert_preserves_chunk_id(self):
        from src.ingestion.storage.vector_upserter import VectorUpserter
        mock_store = MagicMock()
        with patch("src.ingestion.storage.vector_upserter.VectorStoreFactory") as mock_factory:
            mock_factory.create.return_value = mock_store
            upserter = VectorUpserter(MagicMock())

        chunks = [
            _make_chunk("my_id_001", "hello"),
            _make_chunk("my_id_002", "world", chunk_index=1),
        ]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        ids = upserter.upsert(chunks, vectors)
        self.assertEqual(ids, ["my_id_001", "my_id_002"])
        records = mock_store.upsert.call_args[0][0]
        self.assertEqual(records[0]["id"], "my_id_001")

    def test_legacy_generate_chunk_id_compatibility(self):
        from src.ingestion.storage.vector_upserter import VectorUpserter
        self.assertTrue(hasattr(VectorUpserter, "_generate_chunk_id"))

    def test_legacy_upsert_batch_compatibility(self):
        from src.ingestion.storage.vector_upserter import VectorUpserter
        self.assertTrue(hasattr(VectorUpserter, "upsert_batch"))


# ====================================================================
# R5: Parent-Child collision-free indexing
# ====================================================================

class TestR5ParentChildCollision(TestCase):

    def test_no_collision_many_children(self):
        from src.ingestion.chunking.document_chunker import DocumentChunker
        from src.core.types import Document

        chunker = DocumentChunker.__new__(DocumentChunker)
        chunker._parent_child = True
        chunker._parent_size = 100
        chunker._parent_overlap = 10
        splitter = MagicMock()
        splitter.split_text.return_value = [f"child_{i}" for i in range(5)]
        chunker._splitter = splitter

        doc = Document(id="doc1", text="a " * 200, metadata={"source_path": "test.pdf"})
        chunks = chunker.split_document(doc)
        ids = [c.id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate chunk IDs detected")

    def test_child_id_format(self):
        from src.ingestion.chunking.document_chunker import DocumentChunker
        chunker = DocumentChunker.__new__(DocumentChunker)
        child_id = chunker._generate_chunk_id("doc1", "2_5", "some text", prefix="child")
        self.assertIn("child_2_5_", child_id)


# ====================================================================
# R6: BM25 Incremental add_document
# ====================================================================

class TestR6BM25IncrementalAdd(TestCase):

    def test_add_document_incremental(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        idx = BM25Indexer(index_dir=tempfile.mkdtemp())
        batch1 = [{"chunk_id": "a_0", "term_frequencies": {"tcp": 2}, "doc_length": 2}]
        idx.build(batch1, collection="test")
        batch2 = [{"chunk_id": "b_0", "term_frequencies": {"tcp": 1, "udp": 3}, "doc_length": 4}]
        idx.add_document(batch2, collection="test")
        self.assertEqual(idx._metadata["num_docs"], 2)
        self.assertIn("udp", idx._index)
        self.assertEqual(idx._index["tcp"]["df"], 2)

    def test_add_document_empty(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        idx = BM25Indexer(index_dir=tempfile.mkdtemp())
        idx.add_document([], collection="test")

    def test_add_document_no_prior_index(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        idx = BM25Indexer(index_dir=tempfile.mkdtemp())
        batch = [{"chunk_id": "c_0", "term_frequencies": {"hello": 1}, "doc_length": 1}]
        idx.add_document(batch, collection="fresh")
        self.assertEqual(idx._metadata["num_docs"], 1)


# ====================================================================
# R8: Parallel Dense/Sparse encoding
# ====================================================================

class TestR8ParallelEncoding(TestCase):

    def test_dense_sparse_run_in_parallel(self):
        from src.ingestion.embedding.batch_processor import BatchProcessor
        dense = MagicMock()
        sparse = MagicMock()
        dense.encode.return_value = [[0.1, 0.2]]
        sparse.encode.return_value = [{"chunk_id": "c1", "term_frequencies": {"x": 1}, "doc_length": 1, "unique_terms": 1}]
        bp = BatchProcessor(dense_encoder=dense, sparse_encoder=sparse, batch_size=10)
        chunks = [_make_chunk("c1", "hello")]
        result = bp.process(chunks)
        self.assertEqual(result.successful_chunks, 1)
        self.assertEqual(len(result.dense_vectors), 1)
        self.assertEqual(len(result.sparse_stats), 1)

    def test_dense_fails_sparse_still_runs(self):
        from src.ingestion.embedding.batch_processor import BatchProcessor
        dense = MagicMock()
        sparse = MagicMock()
        dense.encode.side_effect = RuntimeError("API error")
        sparse.encode.return_value = [{"chunk_id": "c1", "term_frequencies": {}, "doc_length": 0, "unique_terms": 0}]
        bp = BatchProcessor(dense_encoder=dense, sparse_encoder=sparse, batch_size=10)
        chunks = [_make_chunk("c1", "hello")]
        result = bp.process(chunks)
        self.assertEqual(result.failed_chunks, 1)
        self.assertEqual(len(result.dense_vectors), 0)


# ====================================================================
# R9: Thread-safe LLM lazy loading
# ====================================================================

class TestR9ThreadSafeLLM(TestCase):

    def test_chunk_refiner_has_lock(self):
        from src.ingestion.transform.chunk_refiner import ChunkRefiner
        settings = MagicMock()
        settings.ingestion = None
        refiner = ChunkRefiner(settings)
        self.assertIsInstance(refiner._llm_lock, type(threading.Lock()))

    def test_metadata_enricher_has_lock(self):
        from src.ingestion.transform.metadata_enricher import MetadataEnricher
        settings = MagicMock()
        settings.ingestion = None
        enricher = MetadataEnricher(settings)
        self.assertIsInstance(enricher._llm_lock, type(threading.Lock()))


# ====================================================================
# R10: ContextualEnricher event loop safety
# ====================================================================

class TestR10EventLoopSafety(TestCase):

    def test_rule_mode_works(self):
        from src.ingestion.transform.contextual_enricher import ContextualEnricher
        ce = ContextualEnricher(mode="rule")
        chunks = [_make_chunk("c1", "TCP三次握手", section_title="第三章")]
        result = ce.enrich(chunks, doc_title="计算机网络")
        self.assertIn("计算机网络", result[0].text)

    def test_llm_mode_no_new_event_loop(self):
        import inspect
        from src.ingestion.transform.contextual_enricher import ContextualEnricher
        source = inspect.getsource(ContextualEnricher)
        self.assertNotIn("new_event_loop", source)


# ====================================================================
# R12: SimHash LSH bucketing dedup
# ====================================================================

class TestR12SimHashLSH(TestCase):

    def test_dedup_removes_near_duplicates(self):
        from src.ingestion.transform.chunk_dedup import dedup_chunks
        c1 = _make_chunk("1", "TCP协议是传输控制协议的缩写和说明文档")
        c2 = _make_chunk("2", "TCP协议是传输控制协议的简称和说明文档")
        c3 = _make_chunk("3", "UDP是用户数据报协议完全不同的内容")
        result = dedup_chunks([c1, c2, c3], threshold=5)
        ids = [c.id for c in result]
        self.assertIn("1", ids)
        self.assertIn("3", ids)

    def test_dedup_preserves_order(self):
        from src.ingestion.transform.chunk_dedup import dedup_chunks
        chunks = [_make_chunk(str(i), f"unique content {i} " * 20) for i in range(10)]
        result = dedup_chunks(chunks, threshold=3)
        self.assertEqual(len(result), 10)

    def test_dedup_empty_input(self):
        from src.ingestion.transform.chunk_dedup import dedup_chunks
        self.assertEqual(dedup_chunks([], threshold=3), [])

    def test_no_duplicate_tokenize_function(self):
        import inspect
        from src.ingestion.transform import chunk_dedup
        source = inspect.getsource(chunk_dedup)
        count = source.count("def _tokenize(")
        self.assertEqual(count, 1, f"_tokenize defined {count} times")

    def test_band_keys_cover_all_bits(self):
        from src.ingestion.transform.chunk_dedup import _band_keys, HASH_BITS
        fp = (1 << HASH_BITS) - 1
        keys = _band_keys(fp, num_bands=4)
        self.assertEqual(len(keys), 4)


# ====================================================================
# R13: Trace text trimming
# ====================================================================

class TestR13TraceTrimming(TestCase):

    def test_trace_uses_preview_not_full_text(self):
        import inspect
        from src.ingestion import pipeline
        source = inspect.getsource(pipeline)
        self.assertIn("text_preview", source)


# ====================================================================
# R14: BM25 gzip compression
# ====================================================================

class TestR14BM25GzipCompression(TestCase):

    def test_save_creates_gzip_file(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        tmp = tempfile.mkdtemp()
        idx = BM25Indexer(index_dir=tmp)
        stats = [{"chunk_id": "a", "term_frequencies": {"x": 1}, "doc_length": 1}]
        idx.build(stats, collection="test")
        gz_path = Path(tmp) / "test_bm25.json.gz"
        self.assertTrue(gz_path.exists())
        with gzip.open(gz_path, 'rt') as f:
            data = json.load(f)
        self.assertIn("metadata", data)

    def test_load_gzip_roundtrip(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        tmp = tempfile.mkdtemp()
        idx = BM25Indexer(index_dir=tmp)
        stats = [{"chunk_id": "a", "term_frequencies": {"hello": 2}, "doc_length": 3}]
        idx.build(stats, collection="rt")
        idx2 = BM25Indexer(index_dir=tmp)
        self.assertTrue(idx2.load(collection="rt"))
        self.assertIn("hello", idx2._index)

    def test_load_legacy_json_fallback(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        tmp = tempfile.mkdtemp()
        legacy_path = Path(tmp) / "legacy_bm25.json"
        data = {"metadata": {"num_docs": 1}, "index": {"foo": {"idf": 1.0, "df": 1, "postings": []}}}
        with open(legacy_path, 'w') as f:
            json.dump(data, f)
        idx = BM25Indexer(index_dir=tmp)
        self.assertTrue(idx.load(collection="legacy"))
        self.assertIn("foo", idx._index)


# ====================================================================
# R15: DenseEncoder retry with backoff
# ====================================================================

class TestR15DenseEncoderRetry(TestCase):

    def test_retry_on_transient_failure(self):
        from src.ingestion.embedding.dense_encoder import DenseEncoder
        mock_embedding = MagicMock()
        call_count = 0

        def side_effect(texts, trace=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return [[0.1, 0.2]]

        mock_embedding.embed.side_effect = side_effect
        encoder = DenseEncoder(mock_embedding, batch_size=10, max_retries=2)
        chunks = [_make_chunk("c1", "hello world")]
        vectors = encoder.encode(chunks)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(call_count, 2)

    def test_exhausted_retries_raises(self):
        from src.ingestion.embedding.dense_encoder import DenseEncoder
        mock_embedding = MagicMock()
        mock_embedding.embed.side_effect = RuntimeError("permanent")
        encoder = DenseEncoder(mock_embedding, batch_size=10, max_retries=1)
        chunks = [_make_chunk("c1", "hello world")]
        with self.assertRaises(RuntimeError):
            encoder.encode(chunks)
        self.assertEqual(mock_embedding.embed.call_count, 2)

    def test_no_retry_on_success(self):
        from src.ingestion.embedding.dense_encoder import DenseEncoder
        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1]]
        encoder = DenseEncoder(mock_embedding, batch_size=10, max_retries=2)
        chunks = [_make_chunk("c1", "hello")]
        encoder.encode(chunks)
        self.assertEqual(mock_embedding.embed.call_count, 1)


# ====================================================================
# R7: Stale chunk cleanup on force re-ingest
# ====================================================================

class TestR7StaleChunkCleanup(TestCase):

    def test_cleanup_stale_data_method_exists(self):
        from src.ingestion.pipeline import IngestionPipeline
        self.assertTrue(hasattr(IngestionPipeline, "_cleanup_stale_data"))

    @patch("src.ingestion.pipeline.ImageStorage")
    @patch("src.ingestion.pipeline.create_sparse_index")
    @patch("src.ingestion.pipeline.VectorUpserter")
    @patch("src.ingestion.pipeline.BatchProcessor")
    @patch("src.ingestion.pipeline.ImageCaptioner")
    @patch("src.ingestion.pipeline.MetadataEnricher")
    @patch("src.ingestion.pipeline.ChunkRefiner")
    @patch("src.ingestion.pipeline.DocumentChunker")
    @patch("src.ingestion.pipeline.PdfLoader")
    @patch("src.ingestion.pipeline.PptxLoader")
    @patch("src.ingestion.pipeline.create_ingestion_backends")
    @patch("src.ingestion.pipeline.EmbeddingFactory")
    @patch("src.ingestion.pipeline.VectorStoreFactory")
    @patch("src.ingestion.pipeline.DenseEncoder")
    @patch("src.ingestion.pipeline.SparseEncoder")
    @patch("src.ingestion.pipeline.ContextualEnricher")
    def test_cleanup_calls_vector_store(self, *mocks):
        from src.ingestion.pipeline import IngestionPipeline
        settings = MagicMock()
        settings.ingestion = MagicMock()
        settings.ingestion.batch_size = 10
        settings.ingestion.chunk_refiner = {}
        settings.embedding = MagicMock()
        settings.embedding.provider = "mock"
        settings.vector_store = MagicMock()
        settings.vector_store.provider = "mock"
        settings.retrieval = MagicMock()
        settings.retrieval.contextual_enrichment = "rule"
        settings.retrieval.dedup_enabled = False

        backends = MagicMock()
        backends.object_store = MagicMock()
        backends.integrity_checker = MagicMock()
        backends.document_registry = MagicMock()
        backends.task_store = MagicMock()
        backends.image_storage = MagicMock()
        mocks[5].return_value = backends

        pipeline = IngestionPipeline(settings, collection="test", force=True)
        pipeline._cleanup_stale_data("abc123", "/path/file.pdf")
        pipeline.vector_upserter.vector_store.delete_by_metadata.assert_called_once()
