"""Phase T unit tests — Conflict Detection + Concurrency Safety.

Coverage:
  T1-01..T1-12: Conflict detection types, strategies, detector, resolver
  T2-01..T2-18: Concurrency safety fixes (P0/P1/P2)
"""

import asyncio
import math
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.conflict.types import Conflict, ConflictReport, ConflictType
from src.core.types import RetrievalResult


def _make_result(cid: str, text: str, score: float = 0.8, embedding=None) -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, text=text, score=score, metadata={"source_path": "test"}, embedding=embedding)


# ──────────────────────────────────────────────────────────────────────
# T1: Conflict Detection
# ──────────────────────────────────────────────────────────────────────

class TestT1ConflictTypes(unittest.TestCase):
    def test_conflict_type_values(self):
        self.assertEqual(ConflictType.FACTUAL, "factual")
        self.assertEqual(ConflictType.NUMERICAL, "numerical")
        self.assertEqual(ConflictType.DEFINITIONAL, "definitional")
        self.assertEqual(ConflictType.TEMPORAL, "temporal")

    def test_conflict_model(self):
        c = Conflict(
            type=ConflictType.NUMERICAL,
            chunk_a_id="a", chunk_b_id="b",
            claim_a="x=1", claim_b="x=2",
            confidence=0.8,
        )
        self.assertEqual(c.type, ConflictType.NUMERICAL)

    def test_report_has_conflicts(self):
        empty = ConflictReport()
        self.assertFalse(empty.has_conflicts)

        report = ConflictReport(conflicts=[
            Conflict(type=ConflictType.FACTUAL, chunk_a_id="a", chunk_b_id="b",
                     claim_a="x", claim_b="y", confidence=0.7)
        ])
        self.assertTrue(report.has_conflicts)


class TestT1RuleBasedStrategy(unittest.TestCase):
    def test_numerical_conflict(self):
        from src.core.conflict.strategies.rule_based import RuleBasedStrategy
        s = RuleBasedStrategy()
        ra = _make_result("c1", "TCP默认端口为 80")
        rb = _make_result("c2", "TCP默认端口为 443")
        conflicts = asyncio.run(s.detect("TCP端口", [ra, rb]))
        self.assertTrue(any(c.type == ConflictType.NUMERICAL for c in conflicts))

    def test_definitional_conflict(self):
        from src.core.conflict.strategies.rule_based import RuleBasedStrategy
        s = RuleBasedStrategy()
        ra = _make_result("c1", "路由器是指一种网络层转发设备")
        rb = _make_result("c2", "路由器定义为数据链路层交换机")
        conflicts = asyncio.run(s.detect("路由器", [ra, rb]))
        self.assertTrue(any(c.type == ConflictType.DEFINITIONAL for c in conflicts))

    def test_no_conflict_same_data(self):
        from src.core.conflict.strategies.rule_based import RuleBasedStrategy
        s = RuleBasedStrategy()
        ra = _make_result("c1", "HTTP端口为 80")
        rb = _make_result("c2", "HTTP端口为 80")
        conflicts = asyncio.run(s.detect("HTTP端口", [ra, rb]))
        numerical_conflicts = [c for c in conflicts if c.type == ConflictType.NUMERICAL]
        self.assertEqual(len(numerical_conflicts), 0)


class TestT1EmbeddingSimStrategy(unittest.TestCase):
    def test_high_sim_low_jaccard(self):
        from src.core.conflict.strategies.embedding_sim import EmbeddingSimStrategy
        s = EmbeddingSimStrategy(sim_threshold=0.9, jaccard_ceiling=0.35)
        e1 = [1.0, 0.0, 0.0]
        e2 = [0.99, 0.1, 0.01]
        ra = _make_result("c1", "TCP uses three-way handshake for connection", embedding=e1)
        rb = _make_result("c2", "UDP provides connectionless datagram service", embedding=e2)
        conflicts = asyncio.run(s.detect("protocol", [ra, rb]))
        # cos(e1, e2) ~ 0.994 > 0.9, jaccard is low → should flag
        self.assertTrue(len(conflicts) >= 1)

    def test_skips_when_no_embeddings(self):
        from src.core.conflict.strategies.embedding_sim import EmbeddingSimStrategy
        s = EmbeddingSimStrategy()
        ra = _make_result("c1", "text A")
        rb = _make_result("c2", "text B")
        conflicts = asyncio.run(s.detect("q", [ra, rb]))
        self.assertEqual(len(conflicts), 0)


class TestT1LlmNliStrategy(unittest.TestCase):
    def test_contradiction_detected(self):
        from src.core.conflict.strategies.llm_nli import LLMNliStrategy
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"verdict": "contradiction", "confidence": 0.9, "reason": "互斥"}'
        s = LLMNliStrategy(llm=mock_llm, max_pairs=1)
        ra = _make_result("c1", "A is X")
        rb = _make_result("c2", "A is Y")
        conflicts = asyncio.run(s.detect("query", [ra, rb]))
        self.assertEqual(len(conflicts), 1)
        self.assertAlmostEqual(conflicts[0].confidence, 0.9)

    def test_neutral_no_conflict(self):
        from src.core.conflict.strategies.llm_nli import LLMNliStrategy
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"verdict": "neutral", "confidence": 0.3, "reason": ""}'
        s = LLMNliStrategy(llm=mock_llm, max_pairs=1)
        ra = _make_result("c1", "text A")
        rb = _make_result("c2", "text B")
        conflicts = asyncio.run(s.detect("q", [ra, rb]))
        self.assertEqual(len(conflicts), 0)

    def test_no_llm_returns_empty(self):
        from src.core.conflict.strategies.llm_nli import LLMNliStrategy
        s = LLMNliStrategy(llm=None)
        conflicts = asyncio.run(s.detect("q", [_make_result("c1", "x"), _make_result("c2", "y")]))
        self.assertEqual(len(conflicts), 0)


class TestT1ConflictDetector(unittest.TestCase):
    def test_orchestrates_strategies(self):
        from src.core.conflict.detector import ConflictDetector
        mock_strategy = MagicMock()
        mock_strategy.detect = AsyncMock(return_value=[
            Conflict(type=ConflictType.FACTUAL, chunk_a_id="a", chunk_b_id="b",
                     claim_a="x", claim_b="y", confidence=0.8)
        ])
        d = ConflictDetector(strategies=[mock_strategy])
        ra = _make_result("a", "text A")
        rb = _make_result("b", "text B")
        report = asyncio.run(d.detect("q", [ra, rb]))
        self.assertTrue(report.has_conflicts)
        self.assertEqual(len(report.conflicts), 1)

    def test_deduplicates_conflicts(self):
        from src.core.conflict.detector import ConflictDetector
        c1 = Conflict(type=ConflictType.FACTUAL, chunk_a_id="a", chunk_b_id="b",
                      claim_a="x", claim_b="y", confidence=0.8)
        c2 = Conflict(type=ConflictType.NUMERICAL, chunk_a_id="b", chunk_b_id="a",
                      claim_a="1", claim_b="2", confidence=0.7)
        deduped = ConflictDetector._deduplicate([c1, c2])
        self.assertEqual(len(deduped), 1)

    def test_single_result_no_detection(self):
        from src.core.conflict.detector import ConflictDetector
        d = ConflictDetector()
        report = asyncio.run(d.detect("q", [_make_result("a", "text")]))
        self.assertFalse(report.has_conflicts)


class TestT1ConflictResolver(unittest.TestCase):
    def test_resolve_produces_report(self):
        from src.core.conflict.resolver import ConflictResolver
        r = ConflictResolver()
        conflicts = [
            Conflict(type=ConflictType.NUMERICAL, chunk_a_id="a", chunk_b_id="b",
                     claim_a="1", claim_b="2", confidence=0.8)
        ]
        results = [_make_result("a", "x", score=0.9), _make_result("b", "y", score=0.7), _make_result("c", "z", score=0.6)]
        report = r.resolve(conflicts, results)
        self.assertTrue(report.has_conflicts)
        self.assertIn("c", report.trusted_chunk_ids)
        self.assertIn("1", report.resolution_summary)

    def test_empty_conflicts(self):
        from src.core.conflict.resolver import ConflictResolver
        r = ConflictResolver()
        report = r.resolve([], [_make_result("a", "x")])
        self.assertFalse(report.has_conflicts)


class TestT1FromConfig(unittest.TestCase):
    def test_factory_default(self):
        from src.core.conflict.detector import ConflictDetector
        d = ConflictDetector.from_config()
        self.assertEqual(len(d._strategies), 2)  # rule + embedding

    def test_factory_all_enabled(self):
        from src.core.conflict.detector import ConflictDetector
        mock_llm = MagicMock()
        d = ConflictDetector.from_config(llm=mock_llm, enable_nli=True)
        self.assertEqual(len(d._strategies), 3)


# ──────────────────────────────────────────────────────────────────────
# T2: Concurrency Safety
# ──────────────────────────────────────────────────────────────────────

class TestT2C1ConnLock(unittest.TestCase):
    """C1: All memory modules should have _conn_lock on _get_conn."""

    def test_knowledge_map_has_conn_lock(self):
        from src.agent.memory.knowledge_map import KnowledgeMapMemory
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            m = KnowledgeMapMemory(db_dir=d)
            self.assertIsInstance(m._conn_lock, asyncio.Lock)

    def test_error_memory_has_conn_lock(self):
        from src.agent.memory.error_memory import ErrorMemory
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            m = ErrorMemory(db_dir=d)
            self.assertIsInstance(m._conn_lock, asyncio.Lock)

    def test_session_memory_has_conn_lock(self):
        from src.agent.memory.session_memory import SessionMemory
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            m = SessionMemory(db_dir=d)
            self.assertIsInstance(m._conn_lock, asyncio.Lock)

    def test_student_profile_has_conn_lock(self):
        from src.agent.memory.student_profile import StudentProfileMemory
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            m = StudentProfileMemory(db_dir=d)
            self.assertIsInstance(m._conn_lock, asyncio.Lock)


class TestT2C2BM25Lock(unittest.TestCase):
    """C2: BM25Indexer should have threading.Lock."""

    def test_bm25_has_lock(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            indexer = BM25Indexer(index_dir=d)
            self.assertIsInstance(indexer._lock, threading.Lock)


class TestT2C3SkillMemoryAsync(unittest.TestCase):
    """C3: SkillMemory should use aiosqlite."""

    def test_skill_memory_is_async(self):
        from src.agent.memory.skill_memory import SkillMemory
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sm = SkillMemory(db_dir=d)
            self.assertTrue(hasattr(sm, '_conn_lock'))
            self.assertTrue(asyncio.iscoroutinefunction(sm.save_usage))
            self.assertTrue(asyncio.iscoroutinefunction(sm.search_similar))

    def test_skill_memory_save_and_search(self):
        from src.agent.memory.skill_memory import SkillMemory, ToolUsageRecord
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            sm = SkillMemory(db_dir=d)
            record = ToolUsageRecord(question_pattern="TCP 三次握手", tool_chain=["knowledge_query"])

            async def run():
                await sm.save_usage("u1", record)
                results = await sm.search_similar("u1", "TCP 三次握手")
                return results

            results = asyncio.run(run())
            self.assertGreaterEqual(len(results), 1)


class TestT2C4FileWriteLock(unittest.TestCase):
    """C4: TraceCollector, FileAuditLogger, MetricsCollector should have write locks."""

    def test_trace_collector_has_lock(self):
        from src.core.trace.trace_collector import TraceCollector
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            tc = TraceCollector(traces_path=f"{d}/traces.jsonl")
            self.assertIsInstance(tc._write_lock, threading.Lock)

    def test_audit_logger_has_lock(self):
        from src.agent.hooks.audit import FileAuditLogger
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            al = FileAuditLogger(log_path=f"{d}/audit.jsonl")
            self.assertIsInstance(al._write_lock, threading.Lock)

    def test_metrics_collector_has_lock(self):
        from src.agent.hooks.observability import MetricsCollector
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            mc = MetricsCollector(metrics_path=f"{d}/metrics.jsonl")
            self.assertIsInstance(mc._lock, threading.Lock)


class TestT2C5RateLimitLock(unittest.TestCase):
    """C5: RateLimitHook._buckets should be protected."""

    def test_has_lock(self):
        from src.agent.hooks.rate_limit import RateLimitHook
        hook = RateLimitHook()
        self.assertIsInstance(hook._buckets_lock, threading.Lock)


class TestT2C6CircuitBreakerLock(unittest.TestCase):
    """C6: CircuitBreaker state changes should be atomic."""

    def test_has_lock(self):
        from src.agent.hooks.rate_limit import CircuitBreaker
        cb = CircuitBreaker()
        self.assertIsInstance(cb._lock, threading.Lock)

    def test_thread_safe_transitions(self):
        from src.agent.hooks.rate_limit import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=0.01)
        errors = []

        def hammer():
            try:
                for _ in range(50):
                    cb.record_failure()
                    cb.state
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


class TestT2C7FileConvStoreLockGuard(unittest.TestCase):
    """C7: FileConversationStore._write_locks dict should be guarded."""

    def test_has_locks_guard(self):
        from src.agent.conversation import FileConversationStore
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            store = FileConversationStore(base_dir=d)
            self.assertIsInstance(store._locks_guard, asyncio.Lock)


class TestT2C8MemoryConvStoreLock(unittest.TestCase):
    """C8: MemoryConversationStore._store should be locked."""

    def test_has_store_lock(self):
        from src.agent.conversation import MemoryConversationStore
        store = MemoryConversationStore()
        self.assertIsInstance(store._store_lock, asyncio.Lock)

    def test_concurrent_create(self):
        from src.agent.conversation import MemoryConversationStore

        async def run():
            store = MemoryConversationStore()
            tasks = [store.create(f"user_{i}") for i in range(20)]
            convs = await asyncio.gather(*tasks)
            self.assertEqual(len(convs), 20)
            ids = {c.id for c in convs}
            self.assertEqual(len(ids), 20)

        asyncio.run(run())


class TestT2C9MetricsCounterLock(unittest.TestCase):
    """C9: MetricsCollector counters should be thread-safe."""

    def test_concurrent_counters(self):
        from src.agent.hooks.observability import MetricsCollector
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            mc = MetricsCollector(metrics_path=f"{d}/m.jsonl")

            def inc():
                for _ in range(100):
                    mc.record_counter("test", 1.0)

            threads = [threading.Thread(target=inc) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(mc._counters["test"], 400.0)


class TestT2C10AuditTimesLock(unittest.TestCase):
    """C10: AuditHook._start_times should be locked."""

    def test_has_times_lock(self):
        from src.agent.hooks.audit import AuditHook
        hook = AuditHook()
        self.assertIsInstance(hook._times_lock, threading.Lock)


class TestT2C15IndexCacheInit(unittest.TestCase):
    """C15: SparseRetriever._index_cache should be init'd in __init__."""

    def test_has_index_cache(self):
        from src.core.query_engine.sparse_retriever import SparseRetriever
        sr = SparseRetriever()
        self.assertIsInstance(sr._index_cache, dict)


class TestT2C16TTLCacheLock(unittest.TestCase):
    """C16: TTLCache should be thread-safe."""

    def test_has_lock(self):
        from src.agent.utils.ttl_cache import TTLCache
        cache = TTLCache()
        self.assertTrue(hasattr(cache._lock, 'acquire'))

    def test_concurrent_put_get(self):
        from src.agent.utils.ttl_cache import TTLCache
        cache = TTLCache(max_size=100, ttl_seconds=10)
        errors = []

        def worker(offset):
            try:
                for i in range(100):
                    cache.put(f"key_{offset}_{i}", i)
                    cache.get_value(f"key_{offset}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(j,)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


class TestT2C17CachedEmbeddingLock(unittest.TestCase):
    """C17: CachedEmbedding should have threading.Lock."""

    def test_has_lock(self):
        from src.libs.embedding.cached_embedding import CachedEmbedding
        mock_delegate = MagicMock()
        mock_delegate.embed.return_value = [[0.1, 0.2]]
        ce = CachedEmbedding(delegate=mock_delegate)
        self.assertIsInstance(ce._lock, threading.Lock)


class TestT2C13OffloadAtomicWrite(unittest.TestCase):
    """C13: _offload_to_file should use atomic write."""

    def test_offload_uses_replace(self):
        from src.agent.memory.context_filter import ContextEngineeringFilter
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            f = ContextEngineeringFilter(offload_dir=d)
            ref_id = f._offload_to_file("test content 12345")
            self.assertTrue(len(ref_id) > 0)
            content = f.load_offloaded(ref_id)
            self.assertEqual(content, "test content 12345")


if __name__ == "__main__":
    unittest.main()
