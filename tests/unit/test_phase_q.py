"""Phase Q — RAG recall strategy optimization tests.

Validates all five optimization tracks:
  Q1: Reranker wiring in app.py (_build_hybrid_search)
  Q2: Configuration-driven query enhancement gating
  Q3: Conversation-aware query rewriting
  Q4: MMR embedding reuse (DenseRetriever → HybridSearch)
  Q5: Retrieval quality observability (trace + eval metrics)
"""

from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.types import RetrievalResult


# ===================================================================
# Q1: Reranker wiring
# ===================================================================


class TestQ1RerankerWiring:
    """Verify that _build_hybrid_search creates and injects a reranker."""

    def test_reranker_factory_called_in_build(self):
        """RerankerFactory.create should be called and its result passed to HybridSearch."""
        settings = SimpleNamespace(
            retrieval=SimpleNamespace(
                embedding_cache_size=128,
                rrf_k=60,
                dense_top_k=10,
                sparse_top_k=10,
            ),
            embedding=SimpleNamespace(),
            rerank=SimpleNamespace(enabled=True, provider="llm"),
        )

        fake_reranker = MagicMock()

        with patch("src.core.settings.load_settings", return_value=settings), \
             patch("src.core.settings.resolve_path", return_value="/tmp/bm25"), \
             patch("src.libs.reranker.reranker_factory.RerankerFactory.create", return_value=fake_reranker) as mock_rf, \
             patch("src.libs.embedding.embedding_factory.EmbeddingFactory.create", return_value=MagicMock()), \
             patch("src.libs.vector_store.vector_store_factory.VectorStoreFactory.create", return_value=MagicMock()), \
             patch("src.ingestion.storage.bm25_indexer.BM25Indexer", return_value=MagicMock()), \
             patch("src.core.query_engine.dense_retriever.create_dense_retriever", return_value=MagicMock()), \
             patch("src.core.query_engine.sparse_retriever.create_sparse_retriever") as mock_sp, \
             patch("src.core.query_engine.hybrid_search.HybridSearch") as mock_hs, \
             patch("src.libs.embedding.cached_embedding.CachedEmbedding", side_effect=lambda e, **kw: e):
            mock_sp.return_value = MagicMock()

            from src.server.app import _build_hybrid_search
            _build_hybrid_search("test_col")

            mock_rf.assert_called_once_with(settings)
            mock_hs.assert_called_once()
            call_kwargs = mock_hs.call_args
            assert call_kwargs.kwargs.get("reranker") is fake_reranker or \
                   (call_kwargs.args and fake_reranker in call_kwargs.args)

    def test_reranker_failure_graceful(self):
        """When RerankerFactory.create raises, HybridSearch should still be created."""
        settings = SimpleNamespace(
            retrieval=SimpleNamespace(
                embedding_cache_size=128, rrf_k=60,
                dense_top_k=10, sparse_top_k=10,
            ),
            embedding=SimpleNamespace(),
            rerank=SimpleNamespace(enabled=True, provider="llm"),
        )

        with patch("src.core.settings.load_settings", return_value=settings), \
             patch("src.core.settings.resolve_path", return_value="/tmp/bm25"), \
             patch("src.libs.reranker.reranker_factory.RerankerFactory.create", side_effect=RuntimeError("unavailable")), \
             patch("src.libs.embedding.embedding_factory.EmbeddingFactory.create", return_value=MagicMock()), \
             patch("src.libs.vector_store.vector_store_factory.VectorStoreFactory.create", return_value=MagicMock()), \
             patch("src.ingestion.storage.bm25_indexer.BM25Indexer", return_value=MagicMock()), \
             patch("src.core.query_engine.dense_retriever.create_dense_retriever", return_value=MagicMock()), \
             patch("src.core.query_engine.sparse_retriever.create_sparse_retriever") as mock_sp, \
             patch("src.core.query_engine.hybrid_search.HybridSearch") as mock_hs, \
             patch("src.libs.embedding.cached_embedding.CachedEmbedding", side_effect=lambda e, **kw: e):
            mock_sp.return_value = MagicMock()

            from src.server.app import _build_hybrid_search
            result = _build_hybrid_search("test")
            assert result is not None
            call_kwargs = mock_hs.call_args
            assert call_kwargs.kwargs.get("reranker") is None

    def test_settings_rerank_enabled(self):
        """settings.yaml rerank section should be parseable."""
        from src.core.settings import load_settings
        settings = load_settings()
        rerank = getattr(settings, "rerank", None)
        if rerank is not None:
            assert getattr(rerank, "enabled", False) is True
            assert getattr(rerank, "provider", "") == "llm"


# ===================================================================
# Q2: Configuration-driven query enhancement
# ===================================================================


class TestQ2ConfigDrivenEnhancement:
    """Verify enhancement steps are gated by settings flags."""

    def _make_settings(self, rewrite=False, hyde=False, multi=False):
        return SimpleNamespace(
            retrieval=SimpleNamespace(
                query_rewrite_enabled=rewrite,
                hyde_enabled=hyde,
                multi_query_enabled=multi,
            ),
        )

    def test_flags_read_from_settings(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool

        s_on = self._make_settings(rewrite=True, hyde=True, multi=True)
        tool_on = KnowledgeQueryTool(settings=s_on, hybrid_search=MagicMock())
        assert tool_on._rewrite_enabled is True
        assert tool_on._hyde_enabled is True
        assert tool_on._multi_query_enabled is True

        s_off = self._make_settings(rewrite=False, hyde=False, multi=False)
        tool_off = KnowledgeQueryTool(settings=s_off, hybrid_search=MagicMock())
        assert tool_off._rewrite_enabled is False
        assert tool_off._hyde_enabled is False
        assert tool_off._multi_query_enabled is False

    def test_flags_default_false_without_settings(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool

        tool = KnowledgeQueryTool(settings=None, hybrid_search=MagicMock())
        assert tool._rewrite_enabled is False
        assert tool._hyde_enabled is False
        assert tool._multi_query_enabled is False

    @pytest.mark.asyncio
    async def test_rewrite_skipped_when_disabled(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=self._make_settings(rewrite=False),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"
        ctx = ToolContext(user_id="u", conversation_id="c")
        args = KnowledgeQueryArgs(query="test", collection="computer_network")

        await tool.execute(ctx, args)
        mock_enhancer.rewrite.assert_not_called()
        mock_enhancer.conversation_aware_rewrite.assert_not_called()

    @pytest.mark.asyncio
    async def test_rewrite_called_when_enabled(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_enhancer.rewrite.return_value = "rewritten query"
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=self._make_settings(rewrite=True),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"
        ctx = ToolContext(user_id="u", conversation_id="c")
        args = KnowledgeQueryArgs(query="test", collection="computer_network")

        await tool.execute(ctx, args)
        mock_enhancer.rewrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_hyde_skipped_when_disabled(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=self._make_settings(hyde=False),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"
        ctx = ToolContext(user_id="u", conversation_id="c")
        args = KnowledgeQueryArgs(query="test", collection="computer_network")

        await tool.execute(ctx, args)
        mock_enhancer.hyde_embed.assert_not_called()

    @pytest.mark.asyncio
    async def test_hyde_called_when_enabled(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_enhancer.hyde_embed.return_value = [0.1, 0.2, 0.3]
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=self._make_settings(hyde=True),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"
        ctx = ToolContext(user_id="u", conversation_id="c")
        args = KnowledgeQueryArgs(query="test", collection="computer_network")

        await tool.execute(ctx, args)
        mock_enhancer.hyde_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_query_skipped_when_disabled(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=self._make_settings(multi=False),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"
        ctx = ToolContext(user_id="u", conversation_id="c")
        args = KnowledgeQueryArgs(query="test", collection="computer_network")

        await tool.execute(ctx, args)
        mock_enhancer.decompose.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_query_called_when_enabled(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_enhancer.decompose.return_value = ["sub1", "sub2"]
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=self._make_settings(multi=True),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"
        ctx = ToolContext(user_id="u", conversation_id="c")
        args = KnowledgeQueryArgs(query="test", collection="computer_network")

        await tool.execute(ctx, args)
        mock_enhancer.decompose.assert_called_once()


# ===================================================================
# Q3: Conversation-aware rewriting
# ===================================================================


class TestQ3ConversationAwareRewrite:
    """Verify conversation history integration for pronoun resolution."""

    def test_tool_context_has_recent_messages(self):
        from src.agent.types import ToolContext

        ctx = ToolContext(user_id="u", conversation_id="c")
        assert hasattr(ctx, "recent_messages")
        assert ctx.recent_messages == []

    def test_tool_context_recent_messages_settable(self):
        from src.agent.types import ToolContext

        msgs = [{"role": "user", "content": "TCP是什么"}]
        ctx = ToolContext(user_id="u", conversation_id="c", recent_messages=msgs)
        assert len(ctx.recent_messages) == 1
        assert ctx.recent_messages[0]["content"] == "TCP是什么"

    @pytest.mark.asyncio
    async def test_conversation_aware_rewrite_called_with_history(self):
        """When recent_messages has >1 entries, should use conversation_aware_rewrite."""
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_enhancer.conversation_aware_rewrite.return_value = "TCP三次握手详细过程"
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=SimpleNamespace(
                retrieval=SimpleNamespace(
                    query_rewrite_enabled=True,
                    hyde_enabled=False,
                    multi_query_enabled=False,
                ),
            ),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"

        history = [
            {"role": "user", "content": "TCP是什么"},
            {"role": "assistant", "content": "TCP是传输控制协议"},
        ]
        ctx = ToolContext(user_id="u", conversation_id="c", recent_messages=history)
        args = KnowledgeQueryArgs(query="它的握手过程呢", collection="computer_network")

        await tool.execute(ctx, args)
        mock_enhancer.conversation_aware_rewrite.assert_called_once()
        mock_enhancer.rewrite.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_to_standard_rewrite(self):
        """When recent_messages is empty or has <=1 entry, should use standard rewrite."""
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_enhancer = AsyncMock()
        mock_enhancer.rewrite.return_value = "rewritten"
        mock_hs = MagicMock()
        mock_hs.search = MagicMock(return_value=[])

        tool = KnowledgeQueryTool(
            settings=SimpleNamespace(
                retrieval=SimpleNamespace(
                    query_rewrite_enabled=True,
                    hyde_enabled=False,
                    multi_query_enabled=False,
                ),
            ),
            hybrid_search=mock_hs,
            query_enhancer=mock_enhancer,
        )
        tool._current_collection = "computer_network"

        ctx = ToolContext(user_id="u", conversation_id="c", recent_messages=[])
        args = KnowledgeQueryArgs(query="test", collection="computer_network")
        await tool.execute(ctx, args)
        mock_enhancer.rewrite.assert_called_once()
        mock_enhancer.conversation_aware_rewrite.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_enhancer_conversation_rewrite_method(self):
        """QueryEnhancer.conversation_aware_rewrite should build prompt and call LLM."""
        from src.core.query_engine.query_enhancer import QueryEnhancer

        fake_llm = AsyncMock()
        fake_response = SimpleNamespace(content="TCP传输控制协议的三次握手过程")
        fake_llm.chat.return_value = fake_response

        enhancer = QueryEnhancer(llm_service=fake_llm)
        history = [
            {"role": "user", "content": "TCP是什么"},
            {"role": "assistant", "content": "TCP是传输控制协议"},
        ]
        result = await enhancer.conversation_aware_rewrite("它的握手过程", history)

        assert result == "TCP传输控制协议的三次握手过程"
        fake_llm.chat.assert_called_once()
        call_args = fake_llm.chat.call_args[0][0]
        assert len(call_args) == 1
        prompt_text = call_args[0]["content"]
        assert "它的握手过程" in prompt_text
        assert "TCP" in prompt_text

    @pytest.mark.asyncio
    async def test_conversation_rewrite_no_llm_returns_original(self):
        """Without LLM, conversation_aware_rewrite should return original query."""
        from src.core.query_engine.query_enhancer import QueryEnhancer

        enhancer = QueryEnhancer(llm_service=None)
        result = await enhancer.conversation_aware_rewrite("test", [{"role": "user", "content": "hi"}])
        assert result == "test"

    @pytest.mark.asyncio
    async def test_conversation_rewrite_empty_history_returns_original(self):
        from src.core.query_engine.query_enhancer import QueryEnhancer

        fake_llm = AsyncMock()
        enhancer = QueryEnhancer(llm_service=fake_llm)
        result = await enhancer.conversation_aware_rewrite("test", [])
        assert result == "test"
        fake_llm.chat.assert_not_called()

    def test_conversation_rewrite_prompt_loaded(self):
        """Prompt should be loaded from file or fall back to default."""
        from src.core.query_engine.query_enhancer import QueryEnhancer

        enhancer = QueryEnhancer()
        assert "{history}" in enhancer._conv_rewrite_prompt
        assert "{query}" in enhancer._conv_rewrite_prompt


# ===================================================================
# Q4: MMR embedding reuse
# ===================================================================


class TestQ4MmrEmbeddingReuse:
    """Verify embedding caching from DenseRetriever through MMR in HybridSearch."""

    def test_retrieval_result_has_embedding_field(self):
        r = RetrievalResult(chunk_id="c1", score=0.9, text="hello")
        assert r.embedding is None

        vec = [0.1, 0.2, 0.3]
        r2 = RetrievalResult(chunk_id="c2", score=0.8, text="world", embedding=vec)
        assert r2.embedding == vec

    def test_retrieval_result_embedding_not_in_repr(self):
        vec = [0.1] * 768
        r = RetrievalResult(chunk_id="c1", score=0.9, text="t", embedding=vec)
        assert "0.1" not in repr(r)

    def test_dense_retriever_propagates_embedding(self):
        """DenseRetriever._transform_results should map embedding field."""
        from src.core.query_engine.dense_retriever import DenseRetriever

        retriever = DenseRetriever(
            embedding_client=MagicMock(),
            vector_store=MagicMock(),
        )
        raw = [
            {
                "id": "c1",
                "score": 0.95,
                "text": "TCP三次握手",
                "metadata": {"source_path": "test.pdf"},
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "id": "c2",
                "score": 0.8,
                "text": "UDP无连接",
                "metadata": {"source_path": "test.pdf"},
            },
        ]
        results = retriever._transform_results(raw)
        assert len(results) == 2
        assert results[0].embedding == [0.1, 0.2, 0.3]
        assert results[1].embedding is None

    def test_mmr_reuses_cached_embeddings(self):
        """_apply_mmr should not re-embed candidates that already have embeddings."""
        from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig

        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.5, 0.5, 0.5]]

        hs = HybridSearch.__new__(HybridSearch)
        hs.config = HybridSearchConfig(mmr_enabled=True, mmr_lambda=0.7)
        hs.embedding_client = mock_embedding

        cached_vec = [0.1, 0.2, 0.3]
        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="cached", embedding=cached_vec),
            RetrievalResult(chunk_id="c2", score=0.8, text="cached2", embedding=[0.4, 0.5, 0.6]),
        ]

        with patch("src.core.query_engine.mmr.mmr_rerank",
                    side_effect=lambda **kw: kw["candidates"][:kw["top_k"]]):
            mmr_results = hs._apply_mmr("query", results, top_k=2, trace=None)

        assert len(mmr_results) == 2
        embed_call_texts = []
        for call in mock_embedding.embed.call_args_list:
            if call.args:
                embed_call_texts.extend(call.args[0])
        assert "cached" not in embed_call_texts
        assert "cached2" not in embed_call_texts

    def test_mmr_embeds_missing_candidates(self):
        """_apply_mmr should embed candidates without cached embeddings."""
        from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig

        mock_embedding = MagicMock()
        mock_embedding.embed.side_effect = [
            [[0.5, 0.5, 0.5]],
            [[0.9, 0.9, 0.9]],
        ]

        hs = HybridSearch.__new__(HybridSearch)
        hs.config = HybridSearchConfig(mmr_enabled=True, mmr_lambda=0.7)
        hs.embedding_client = mock_embedding

        results = [
            RetrievalResult(chunk_id="c1", score=0.9, text="has_embed", embedding=[0.1, 0.2, 0.3]),
            RetrievalResult(chunk_id="c2", score=0.8, text="no_embed"),
        ]

        with patch("src.core.query_engine.mmr.mmr_rerank",
                    side_effect=lambda **kw: kw["candidates"][:kw["top_k"]]):
            hs._apply_mmr("query", results, top_k=2, trace=None)

        embed_calls = mock_embedding.embed.call_args_list
        assert len(embed_calls) == 2
        second_call_texts = embed_calls[1].args[0] if embed_calls[1].args else embed_calls[1].kwargs.get("texts", [])
        assert "no_embed" in second_call_texts


# ===================================================================
# Q5: Retrieval quality observability
# ===================================================================


class TestQ5RetrievalObservability:
    """Verify trace recording and offline evaluation metrics."""

    def test_retrieval_summary_trace_recorded(self):
        """HybridSearch.search should record a 'retrieval_summary' trace stage."""
        from src.core.query_engine.hybrid_search import HybridSearch, HybridSearchConfig

        mock_dense = MagicMock()
        mock_dense.retrieve.return_value = [
            RetrievalResult(chunk_id="c1", score=0.9, text="result", metadata={"source_path": "t"}),
        ]
        mock_sparse = MagicMock()
        mock_sparse.retrieve.return_value = []
        mock_fusion = MagicMock()
        mock_fusion.fuse.return_value = [
            RetrievalResult(chunk_id="c1", score=0.9, text="result", metadata={"source_path": "t"}),
        ]
        mock_fusion.fuse_with_weights.return_value = mock_fusion.fuse.return_value
        mock_processor = MagicMock()
        from src.core.types import ProcessedQuery
        mock_processor.process.return_value = ProcessedQuery(
            original_query="test", keywords=["test"],
        )

        settings = SimpleNamespace(
            retrieval=SimpleNamespace(
                dense_top_k=10, sparse_top_k=10, fusion_top_k=5,
                enable_dense=True, enable_sparse=True,
                dense_weight=1.0, sparse_weight=1.0,
                rerank_enabled=False, rerank_top_k=5,
                mmr_enabled=False, mmr_lambda=0.7,
                min_score=0.0, post_dedup_enabled=False,
                parallel_retrieval=False,
                metadata_filter_post=False,
            )
        )

        hs = HybridSearch(
            settings=settings,
            query_processor=mock_processor,
            dense_retriever=mock_dense,
            sparse_retriever=mock_sparse,
            fusion=mock_fusion,
        )

        mock_trace = MagicMock()
        hs.search(query="test", top_k=5, trace=mock_trace)

        stage_names = [call.args[0] for call in mock_trace.record_stage.call_args_list]
        assert "retrieval_summary" in stage_names

        summary_call = None
        for call in mock_trace.record_stage.call_args_list:
            if call.args[0] == "retrieval_summary":
                summary_call = call.args[1]
                break

        assert summary_call is not None
        assert "total_results" in summary_call
        assert "score_max" in summary_call
        assert "score_min" in summary_call
        assert "score_mean" in summary_call

    def test_eval_hit_rate_computation(self):
        from scripts.eval_retrieval import hit_rate

        assert hit_rate(["a", "b", "c"], ["b"]) == 1.0
        assert hit_rate(["a", "b", "c"], ["d"]) == 0.0
        assert hit_rate([], ["a"]) == 0.0
        assert hit_rate(["a"], []) == 0.0

    def test_eval_reciprocal_rank_computation(self):
        from scripts.eval_retrieval import reciprocal_rank

        assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0
        assert reciprocal_rank(["a", "b", "c"], ["b"]) == pytest.approx(0.5)
        assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
        assert reciprocal_rank(["a", "b", "c"], ["d"]) == 0.0

    def test_eval_ndcg_computation(self):
        from scripts.eval_retrieval import ndcg

        score = ndcg(["a", "b"], ["a"], k=5)
        assert score == pytest.approx(1.0)

        score_miss = ndcg(["x", "y"], ["a"], k=5)
        assert score_miss == 0.0

        score_second = ndcg(["x", "a"], ["a"], k=5)
        expected_dcg = 1.0 / math.log2(3)
        expected_idcg = 1.0 / math.log2(2)
        assert score_second == pytest.approx(expected_dcg / expected_idcg)

    def test_eval_edge_cases(self):
        from scripts.eval_retrieval import hit_rate, reciprocal_rank, ndcg

        assert hit_rate([], []) == 0.0
        assert reciprocal_rank([], []) == 0.0
        assert ndcg([], [], k=5) == 0.0

        assert hit_rate(["a", "b"], ["a", "b"]) == 1.0
        assert reciprocal_rank(["a", "b"], ["a", "b"]) == 1.0

    def test_eval_load_eval_set(self, tmp_path):
        from scripts.eval_retrieval import load_eval_set

        data = tmp_path / "eval.jsonl"
        data.write_text(
            '{"query": "q1", "expected_ids": ["c1"]}\n'
            '{"query": "q2", "expected_ids": ["c2", "c3"]}\n'
        )
        items = load_eval_set(str(data))
        assert len(items) == 2
        assert items[0]["query"] == "q1"
        assert items[1]["expected_ids"] == ["c2", "c3"]
