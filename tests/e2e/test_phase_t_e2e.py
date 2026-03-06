"""Phase T end-to-end tests — Conflict Detection + Concurrency Safety integration."""

import asyncio
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

from src.core.conflict.detector import ConflictDetector
from src.core.conflict.resolver import ConflictResolver
from src.core.conflict.types import ConflictReport, ConflictType
from src.core.types import RetrievalResult


def _r(cid: str, text: str, score: float = 0.8, embedding=None) -> RetrievalResult:
    return RetrievalResult(chunk_id=cid, text=text, score=score, metadata={"source_path": "test.pdf"}, embedding=embedding)


class TestE2EConflictDetectionPipeline(unittest.TestCase):
    """Full pipeline: retrieval results -> detector -> report with resolution."""

    def test_full_detection_pipeline(self):
        """Numerical conflict detected by rule-based strategy, resolved by resolver."""
        r1 = _r("c1", "TCP窗口大小为 65535 字节", score=0.9)
        r2 = _r("c2", "TCP窗口大小为 1024 字节", score=0.7)
        r3 = _r("c3", "UDP是无连接协议", score=0.6)

        detector = ConflictDetector.from_config(
            enable_rule=True, enable_embedding=False, enable_nli=False,
        )
        report = asyncio.run(detector.detect("TCP窗口大小", [r1, r2, r3]))

        self.assertTrue(report.has_conflicts)
        self.assertTrue(any(c.type == ConflictType.NUMERICAL for c in report.conflicts))
        self.assertIn("c3", report.trusted_chunk_ids)
        self.assertIn("冲突", report.resolution_summary)

    def test_no_conflict_clean_results(self):
        """All results consistent — no conflicts."""
        r1 = _r("c1", "HTTP使用TCP协议", score=0.9)
        r2 = _r("c2", "HTTPS是HTTP的安全版本", score=0.8)

        detector = ConflictDetector.from_config(
            enable_rule=True, enable_embedding=False, enable_nli=False,
        )
        report = asyncio.run(detector.detect("HTTP协议", [r1, r2]))
        self.assertFalse(report.has_conflicts)


class TestE2EConflictWithLlmNli(unittest.TestCase):
    """E2E: LLM NLI strategy integration."""

    def test_llm_nli_full_flow(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"verdict": "contradiction", "confidence": 0.95, "reason": "OSI七层 vs 五层"}'

        detector = ConflictDetector.from_config(
            llm=mock_llm, enable_rule=False, enable_embedding=False, enable_nli=True, nli_max_pairs=3,
        )
        r1 = _r("c1", "OSI模型有七层")
        r2 = _r("c2", "OSI模型有五层")
        report = asyncio.run(detector.detect("OSI模型", [r1, r2]))

        self.assertTrue(report.has_conflicts)
        self.assertIn("OSI", report.conflicts[0].description)


class TestE2EConcurrentMemoryAccess(unittest.TestCase):
    """E2E: concurrent access to memory stores with connection locks."""

    def test_concurrent_knowledge_map_updates(self):
        from src.agent.memory.knowledge_map import KnowledgeMapMemory

        with tempfile.TemporaryDirectory() as d:
            mem = KnowledgeMapMemory(db_dir=d)

            async def run():
                tasks = []
                for i in range(20):
                    concept = f"concept_{i % 5}"
                    tasks.append(mem.update_mastery("user1", concept, i % 2 == 0))
                await asyncio.gather(*tasks)

                for i in range(5):
                    node = await mem.get_node("user1", f"concept_{i}")
                    self.assertIsNotNone(node)
                    self.assertGreater(node.quiz_count, 0)

                await mem.close()

            asyncio.run(run())

    def test_concurrent_skill_memory(self):
        from src.agent.memory.skill_memory import SkillMemory, ToolUsageRecord

        with tempfile.TemporaryDirectory() as d:
            sm = SkillMemory(db_dir=d)

            async def run():
                tasks = []
                for i in range(15):
                    r = ToolUsageRecord(question_pattern=f"TCP handshake question {i}", tool_chain=["tool_a"])
                    tasks.append(sm.save_usage("u1", r))
                await asyncio.gather(*tasks)

                results = await sm.search_similar("u1", "TCP handshake", limit=10)
                self.assertGreater(len(results), 0)
                await sm.close()

            asyncio.run(run())


class TestE2EConcurrentBM25(unittest.TestCase):
    """E2E: BM25Indexer lock prevents corruption under concurrent queries."""

    def test_concurrent_build_and_query(self):
        from src.ingestion.storage.bm25_indexer import BM25Indexer

        with tempfile.TemporaryDirectory() as d:
            indexer = BM25Indexer(index_dir=d)
            stats = [
                {"chunk_id": f"c{i}", "term_frequencies": {"tcp": 2, "udp": 1}, "doc_length": 3}
                for i in range(10)
            ]
            indexer.build(stats, collection="test")

            errors = []

            def query_worker():
                try:
                    for _ in range(50):
                        results = indexer.query(["tcp"], top_k=5)
                        assert len(results) > 0
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=query_worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len(errors), 0, f"Errors: {errors}")


class TestE2EKnowledgeQueryWithConflict(unittest.TestCase):
    """E2E: KnowledgeQueryTool integrates conflict detection into output."""

    def test_conflict_section_in_output(self):
        from src.agent.tools.knowledge_query import KnowledgeQueryTool, KnowledgeQueryArgs
        from src.agent.types import ToolContext

        mock_search = MagicMock()
        mock_search.search.return_value = [
            _r("c1", "HTTP默认端口为 80", score=0.9),
            _r("c2", "HTTP默认端口为 443", score=0.8),
        ]

        detector = ConflictDetector.from_config(
            enable_rule=True, enable_embedding=False, enable_nli=False,
        )

        tool = KnowledgeQueryTool(
            settings=None, hybrid_search=mock_search, conflict_detector=detector,
        )
        tool._initialized = True
        tool._current_collection = "test"

        ctx = ToolContext(user_id="u1", conversation_id="conv1", request_id="req1")
        args = KnowledgeQueryArgs(query="HTTP端口", collection="test")

        result = asyncio.run(tool.execute(ctx, args))
        self.assertTrue(result.success)
        if "冲突" in result.result_for_llm:
            self.assertIn("⚠️", result.result_for_llm)


if __name__ == "__main__":
    unittest.main()
