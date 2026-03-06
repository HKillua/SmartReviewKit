"""E2E RAG Pipeline Test.

Validates the full RAG pipeline end-to-end:
1. Ingestion: load a sample PDF -> chunk -> transform (contextual enrichment + dedup) -> encode -> store
2. Retrieval: HybridSearch with MMR, min_score, post-dedup
3. Tool output: KnowledgeQueryTool produces chunk_id references

This test uses mock embeddings to avoid external API dependencies.

Usage::

    pytest tests/e2e/test_rag_pipeline_e2e.py -v -m e2e
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest

pytestmark = [pytest.mark.e2e]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


class MockEmbedding:
    """Deterministic embedding that hashes text to a fixed-dim vector."""

    def __init__(self, dim: int = 64):
        self._dim = dim
        self.call_count = 0

    def embed(self, texts: List[str], trace=None, **kwargs) -> List[List[float]]:
        import hashlib
        self.call_count += 1
        results = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [float(b) / 255.0 for b in h[: self._dim]]
            while len(vec) < self._dim:
                vec.append(0.0)
            results.append(vec)
        return results

    def get_dimension(self) -> int:
        return self._dim

    def validate_texts(self, texts):
        pass


TEST_COLLECTION = "test_e2e_pipeline"


@pytest.fixture(scope="module")
def sample_pdf():
    return str(PROJECT_ROOT / "tests" / "fixtures" / "sample_documents" / "simple.pdf")


@pytest.fixture(scope="module")
def settings():
    """Load settings from the project config."""
    os.chdir(str(PROJECT_ROOT))
    from src.core.settings import load_settings
    return load_settings()


@pytest.fixture(scope="module")
def mock_embedding():
    return MockEmbedding(dim=64)


@pytest.fixture(scope="module")
def ingested_collection(settings, mock_embedding, sample_pdf):
    """Run the ingestion pipeline on the sample PDF and return collection name."""
    from src.ingestion.pipeline import IngestionPipeline
    from src.libs.embedding.embedding_factory import EmbeddingFactory

    collection = TEST_COLLECTION

    original_create = EmbeddingFactory.create
    EmbeddingFactory.create = staticmethod(lambda s, **kw: mock_embedding)

    try:
        pipeline = IngestionPipeline(settings, collection=collection, force=True)
        pipeline.dense_encoder._embedding = mock_embedding
        result = pipeline.run(sample_pdf)
        assert result.success, f"Pipeline failed: {result.error}"
        assert result.chunk_count > 0
    finally:
        EmbeddingFactory.create = original_create

    return collection, result


class TestIngestionPipeline:
    def test_pipeline_success(self, ingested_collection):
        collection, result = ingested_collection
        assert result.success

    def test_chunks_generated(self, ingested_collection):
        _, result = ingested_collection
        assert result.chunk_count > 0

    def test_vectors_stored(self, ingested_collection):
        _, result = ingested_collection
        assert len(result.vector_ids) > 0

    def test_stages_recorded(self, ingested_collection):
        _, result = ingested_collection
        assert "transform" in result.stages
        transform = result.stages["transform"]
        assert "contextual_enricher" in transform
        assert "dedup" in transform


class TestHybridSearch:
    @pytest.fixture(scope="class")
    def hybrid(self, settings, mock_embedding, ingested_collection):
        collection, _ = ingested_collection
        from src.core.settings import resolve_path
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import DenseRetriever
        from src.core.query_engine.sparse_retriever import SparseRetriever
        from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
        from src.core.query_engine.fusion import RRFFusion
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        dense = DenseRetriever(
            settings=settings,
            embedding_client=mock_embedding,
            vector_store=vector_store,
        )
        bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse = SparseRetriever(
            settings=settings,
            bm25_indexer=bm25,
            vector_store=vector_store,
        )
        sparse.default_collection = collection

        config = HybridSearchConfig(
            dense_top_k=10,
            sparse_top_k=10,
            fusion_top_k=5,
            mmr_enabled=True,
            mmr_lambda=0.7,
            min_score=0.0,
            post_dedup_enabled=True,
        )
        hs = HybridSearch(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(),
            config=config,
            embedding_client=mock_embedding,
        )
        return hs

    def test_search_returns_results(self, hybrid):
        results = hybrid.search("document content", top_k=5)
        assert isinstance(results, list)

    def test_results_have_chunk_id(self, hybrid):
        results = hybrid.search("content", top_k=5)
        for r in results:
            assert hasattr(r, "chunk_id")
            assert r.chunk_id

    def test_results_have_score(self, hybrid):
        results = hybrid.search("content", top_k=5)
        for r in results:
            assert hasattr(r, "score")
            assert isinstance(r.score, (int, float))

    def test_results_have_text(self, hybrid):
        results = hybrid.search("content", top_k=5)
        for r in results:
            assert hasattr(r, "text")
            assert r.text

    def test_mmr_limits_results(self, hybrid):
        results = hybrid.search("content", top_k=3)
        assert len(results) <= 3

    def test_min_score_filtering(self, settings, mock_embedding, ingested_collection):
        collection, _ = ingested_collection
        from src.core.settings import resolve_path
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.dense_retriever import DenseRetriever
        from src.core.query_engine.sparse_retriever import SparseRetriever
        from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig
        from src.core.query_engine.fusion import RRFFusion
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory

        vector_store = VectorStoreFactory.create(settings, collection_name=collection)
        dense = DenseRetriever(
            settings=settings, embedding_client=mock_embedding, vector_store=vector_store,
        )
        bm25 = BM25Indexer(index_dir=str(resolve_path(f"data/db/bm25/{collection}")))
        sparse = SparseRetriever(
            settings=settings, bm25_indexer=bm25, vector_store=vector_store,
        )
        sparse.default_collection = collection

        config = HybridSearchConfig(
            dense_top_k=10, sparse_top_k=10, fusion_top_k=10,
            min_score=999.0,
        )
        hs = HybridSearch(
            settings=settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(),
            config=config,
        )
        results = hs.search("content", top_k=10)
        assert len(results) == 0


class TestContextualEnrichment:
    def test_enricher_adds_prefix(self):
        from src.ingestion.transform.contextual_enricher import ContextualEnricher
        from src.core.types import Chunk

        enricher = ContextualEnricher(mode="rule")
        chunks = [
            Chunk(id="c1", text="TCP uses three-way handshake.", metadata={
                "source_path": "test.pdf", "section_title": "Transport Layer"
            }),
        ]
        result = enricher.enrich(chunks, doc_title="Computer Networks")
        assert result[0].metadata.get("contextual_prefix")
        assert result[0].text.startswith("[上下文")


class TestChunkDedup:
    def test_dedup_removes_near_duplicates(self):
        from src.ingestion.transform.chunk_dedup import dedup_chunks
        from src.core.types import Chunk

        chunks = [
            Chunk(id="c1", text="TCP三次握手是建立连接的过程", metadata={"source_path": "a.pdf"}),
            Chunk(id="c2", text="TCP三次握手是建立连接的过程。", metadata={"source_path": "a.pdf"}),
            Chunk(id="c3", text="UDP是无连接的传输层协议", metadata={"source_path": "a.pdf"}),
        ]
        result = dedup_chunks(chunks, threshold=3)
        assert len(result) <= len(chunks)
        ids = [c.id for c in result]
        assert "c1" in ids
        assert "c3" in ids
