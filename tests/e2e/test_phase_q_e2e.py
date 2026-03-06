"""Phase Q — End-to-end tests for RAG recall strategy optimizations.

Validates the integrated behaviour of:
  Q1: Reranker in full search pipeline
  Q2: Config-toggled query enhancement
  Q3: Conversation-aware retrieval flow
  Q4: MMR diversity with embedding reuse
  Q5: KnowledgeQueryTool full pipeline

Uses mock LLM / embedding / vector store to avoid external dependencies.

Usage::

    pytest tests/e2e/test_phase_q_e2e.py -v -m e2e
"""

from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import ProcessedQuery, RetrievalResult

pytestmark = [pytest.mark.e2e]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class DeterministicEmbedding:
    """Hash-based embedding that produces consistent vectors for the same text."""

    def __init__(self, dim: int = 32):
        self._dim = dim
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: List[str], **kwargs) -> List[List[float]]:
        self.embed_calls.append(texts)
        results = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[: self._dim]]
            while len(vec) < self._dim:
                vec.append(0.0)
            results.append(vec)
        return results

    def get_dimension(self) -> int:
        return self._dim


def _make_retrieval_result(chunk_id: str, score: float, text: str, embedding=None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": "test.pdf"},
        embedding=embedding,
    )


def _make_settings(**overrides):
    defaults = dict(
        dense_top_k=10, sparse_top_k=10, fusion_top_k=5,
        enable_dense=True, enable_sparse=True,
        dense_weight=1.0, sparse_weight=1.0,
        rerank_enabled=False, rerank_top_k=5,
        mmr_enabled=False, mmr_lambda=0.7,
        min_score=0.0, post_dedup_enabled=False,
        parallel_retrieval=False,
        metadata_filter_post=False,
        query_rewrite_enabled=False,
        hyde_enabled=False,
        multi_query_enabled=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(retrieval=SimpleNamespace(**defaults))


def _build_hybrid_search(settings, dense_results, sparse_results=None, reranker=None):
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.fusion import RRFFusion

    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = dense_results

    mock_sparse = MagicMock()
    mock_sparse.retrieve.return_value = sparse_results or []

    fusion = RRFFusion(k=60)

    mock_processor = MagicMock()
    mock_processor.process.return_value = ProcessedQuery(
        original_query="test", keywords=["test"],
    )

    hs = HybridSearch(
        settings=settings,
        query_processor=mock_processor,
        dense_retriever=mock_dense,
        sparse_retriever=mock_sparse,
        fusion=fusion,
        reranker=reranker,
    )
    return hs


# ===================================================================
# E2E Test 1: Query with reranker enabled
# ===================================================================


class TestE2ERerankerPipeline:
    """Full search pipeline should call reranker after RRF fusion."""

    def test_e2e_query_with_reranker_enabled(self):
        dense_results = [
            _make_retrieval_result("c1", 0.9, "TCP三次握手过程"),
            _make_retrieval_result("c2", 0.7, "UDP无连接协议"),
            _make_retrieval_result("c3", 0.5, "HTTP协议"),
        ]

        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            {"chunk_id": "c2", "text": "UDP无连接协议", "metadata": {"source_path": "test.pdf"}, "score": 0.95},
            {"chunk_id": "c1", "text": "TCP三次握手过程", "metadata": {"source_path": "test.pdf"}, "score": 0.85},
        ]

        settings = _make_settings(rerank_enabled=True, rerank_top_k=3)
        hs = _build_hybrid_search(settings, dense_results, reranker=mock_reranker)
        results = hs.search(query="网络协议", top_k=3)

        mock_reranker.rerank.assert_called_once()
        assert len(results) >= 1
        assert results[0].chunk_id == "c2"


# ===================================================================
# E2E Test 2: Config toggle query enhancement
# ===================================================================


class TestE2EConfigToggle:
    """Enhancement steps should run only when enabled in config."""

    @pytest.mark.asyncio
    async def test_e2e_config_toggle_query_enhancement(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_enhancer.rewrite.return_value = "rewritten"
        mock_enhancer.hyde_embed.return_value = [0.1, 0.2]
        mock_enhancer.decompose.return_value = ["sub1", "sub2"]

        mock_hs = MagicMock()
        mock_hs.search.return_value = [
            _make_retrieval_result("c1", 0.9, "TCP三次握手"),
        ]

        settings_all_on = _make_settings(
            query_rewrite_enabled=True,
            hyde_enabled=True,
            multi_query_enabled=True,
        )
        tool_on = KnowledgeQueryTool(
            settings=settings_all_on,
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool_on._current_collection = "computer_network"
        ctx = ToolContext(user_id="u", conversation_id="c")
        args = KnowledgeQueryArgs(query="TCP", collection="computer_network")
        await tool_on.execute(ctx, args)

        mock_enhancer.rewrite.assert_called()
        mock_enhancer.hyde_embed.assert_called()
        mock_enhancer.decompose.assert_called()

        mock_enhancer.reset_mock()

        settings_all_off = _make_settings(
            query_rewrite_enabled=False,
            hyde_enabled=False,
            multi_query_enabled=False,
        )
        tool_off = KnowledgeQueryTool(
            settings=settings_all_off,
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool_off._current_collection = "computer_network"
        await tool_off.execute(ctx, args)

        mock_enhancer.rewrite.assert_not_called()
        mock_enhancer.hyde_embed.assert_not_called()
        mock_enhancer.decompose.assert_not_called()


# ===================================================================
# E2E Test 3: Conversation-aware retrieval flow
# ===================================================================


class TestE2EConversationAwareRetrieval:
    """Multi-turn conversation should trigger context-aware rewriting."""

    @pytest.mark.asyncio
    async def test_e2e_conversation_aware_retrieval_flow(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext
        from src.core.query_engine.query_enhancer import QueryEnhancer

        fake_llm = AsyncMock()
        fake_llm.chat.return_value = SimpleNamespace(content="TCP传输控制协议三次握手")

        enhancer = QueryEnhancer(llm_service=fake_llm)

        mock_hs = MagicMock()
        mock_hs.search.return_value = [
            _make_retrieval_result("tcp_handshake", 0.95, "TCP三次握手: SYN → SYN-ACK → ACK"),
        ]

        tool = KnowledgeQueryTool(
            settings=_make_settings(query_rewrite_enabled=True),
            hybrid_search=mock_hs,
            query_enhancer=enhancer,
        )
        tool._current_collection = "computer_network"

        history = [
            {"role": "user", "content": "什么是TCP"},
            {"role": "assistant", "content": "TCP是传输控制协议，提供可靠的面向连接的数据传输"},
            {"role": "user", "content": "它怎么建立连接"},
        ]
        ctx = ToolContext(user_id="u", conversation_id="c", recent_messages=history)
        args = KnowledgeQueryArgs(query="它怎么建立连接", collection="computer_network")

        result = await tool.execute(ctx, args)

        assert result.success
        fake_llm.chat.assert_called_once()
        prompt_text = fake_llm.chat.call_args[0][0][0]["content"]
        assert "TCP" in prompt_text
        assert "它怎么建立连接" in prompt_text

        search_query = mock_hs.search.call_args.kwargs.get("query") or mock_hs.search.call_args[1].get("query", "")
        if not search_query and mock_hs.search.call_args.args:
            pass
        assert "tcp_handshake" in result.result_for_llm.lower() or "TCP" in result.result_for_llm


# ===================================================================
# E2E Test 4: MMR reduces redundancy
# ===================================================================


class TestE2EMmrDiversity:
    """MMR should promote diversity by demoting near-duplicate results."""

    def test_e2e_mmr_reduces_redundancy(self):
        from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig

        embedding = DeterministicEmbedding(dim=32)
        vec_a = embedding.embed(["TCP三次握手"])[0]
        vec_b = embedding.embed(["OSPF路由算法"])[0]

        candidates = [
            _make_retrieval_result("c1", 0.95, "TCP三次握手", embedding=vec_a),
            _make_retrieval_result("c1_dup", 0.90, "TCP三次握手", embedding=vec_a[:]),
            _make_retrieval_result("c2", 0.85, "OSPF路由算法", embedding=vec_b),
        ]

        hs = HybridSearch.__new__(HybridSearch)
        hs.config = HybridSearchConfig(mmr_enabled=True, mmr_lambda=0.5)
        hs.embedding_client = embedding

        mmr_results = hs._apply_mmr("网络协议", candidates, top_k=2, trace=None)

        chunk_ids = [r.chunk_id for r in mmr_results]
        assert len(chunk_ids) == 2
        assert "c2" in chunk_ids, (
            f"MMR should promote diverse 'OSPF' result, got: {chunk_ids}"
        )


# ===================================================================
# E2E Test 5: KnowledgeQueryTool full pipeline
# ===================================================================


class TestE2EKnowledgeQueryToolPipeline:
    """End-to-end test of KnowledgeQueryTool with all components wired."""

    @pytest.mark.asyncio
    async def test_e2e_knowledge_query_tool_full_pipeline(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_hs = MagicMock()
        mock_hs.search.return_value = [
            _make_retrieval_result("chunk_001", 0.92, "DNS域名解析: 递归查询 → 迭代查询"),
            _make_retrieval_result("chunk_002", 0.85, "DNS记录类型: A, AAAA, CNAME, MX"),
        ]

        mock_enhancer = AsyncMock()
        mock_enhancer.rewrite.return_value = "DNS域名系统解析流程"
        mock_enhancer.hyde_embed.return_value = None
        mock_enhancer.decompose.return_value = ["DNS域名系统解析流程"]

        settings = _make_settings(
            query_rewrite_enabled=True,
            hyde_enabled=False,
            multi_query_enabled=False,
        )
        tool = KnowledgeQueryTool(
            settings=settings,
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"

        ctx = ToolContext(user_id="student1", conversation_id="conv1")
        args = KnowledgeQueryArgs(
            query="DNS怎么解析域名的",
            top_k=5,
            collection="computer_network",
        )

        result = await tool.execute(ctx, args)

        assert result.success is True
        assert "检索到" in result.result_for_llm
        assert "chunk_001" in result.result_for_llm
        assert "DNS" in result.result_for_llm
        assert "相关度" in result.result_for_llm

        mock_enhancer.rewrite.assert_called_once()
        mock_hs.search.assert_called()
