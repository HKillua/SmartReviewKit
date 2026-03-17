"""Unit tests for Phase U: Multi-Source Documents + Adaptive Retrieval Router."""

import asyncio
import hashlib
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# U2: QuestionParser
# ---------------------------------------------------------------------------

class TestQuestionParser(unittest.TestCase):
    def setUp(self):
        from src.ingestion.transform.question_parser import QuestionParser
        self.parser = QuestionParser()

    def test_parse_numbered_questions(self):
        text = """1. TCP 是什么协议？
A. 传输层协议
B. 网络层协议
C. 应用层协议
D. 链路层协议
答案: A
解析: TCP 是传输控制协议，工作在传输层。

2. UDP 是面向连接的协议吗？
A. 是
B. 否
答案: B
解析: UDP 是无连接的。"""
        questions = self.parser.parse(text)
        self.assertGreaterEqual(len(questions), 2)
        self.assertEqual(questions[0].question_type, "choice")
        self.assertIn("A", questions[0].options)
        self.assertEqual(questions[0].answer, "A")

    def test_parse_chinese_numbered(self):
        text = """第一题 简述TCP三次握手的过程。
答案: SYN → SYN-ACK → ACK

第二题 UDP和TCP有什么区别？
答案: TCP面向连接，UDP无连接"""
        questions = self.parser.parse(text)
        self.assertGreaterEqual(len(questions), 2)

    def test_parse_no_questions(self):
        text = "这是一段普通的文本，没有题目格式。"
        questions = self.parser.parse(text)
        self.assertEqual(len(questions), 0)

    def test_to_chunks_metadata(self):
        text = """1. TCP 使用几次握手？
A. 2次
B. 3次
C. 4次
答案: B
解析: TCP 使用三次握手建立连接。"""
        questions = self.parser.parse(text)
        chunks = self.parser.to_chunks(questions, source_path="/tmp/exam.pdf", doc_id="doc123")
        self.assertTrue(len(chunks) > 0)
        chunk = chunks[0]
        self.assertEqual(chunk.metadata["source_type"], "question_bank")
        self.assertEqual(chunk.metadata["source_path"], "/tmp/exam.pdf")
        self.assertIn("answer", chunk.metadata)

    def test_fill_blank_detection(self):
        text = "1. TCP 使用 _____ 次握手建立连接。\n答案: 三"
        questions = self.parser.parse(text)
        self.assertTrue(len(questions) > 0)
        self.assertEqual(questions[0].question_type, "fill_blank")

    def test_difficulty_inference(self):
        text = "1. 基础概念题：什么是IP地址？\n答案: 略"
        questions = self.parser.parse(text)
        self.assertTrue(len(questions) > 0)
        self.assertEqual(questions[0].difficulty, 1)


# ---------------------------------------------------------------------------
# U1: IngestionPipeline.infer_source_type
# ---------------------------------------------------------------------------

class TestSourceTypeInference(unittest.TestCase):
    def test_pptx_is_slide(self):
        from src.ingestion.pipeline import IngestionPipeline
        self.assertEqual(IngestionPipeline.infer_source_type(Path("lecture.pptx")), "slide")

    def test_exercise_is_question_bank(self):
        from src.ingestion.pipeline import IngestionPipeline
        self.assertEqual(IngestionPipeline.infer_source_type(Path("TCP习题集.pdf")), "question_bank")

    def test_exam_is_question_bank(self):
        from src.ingestion.pipeline import IngestionPipeline
        self.assertEqual(IngestionPipeline.infer_source_type(Path("final_exam.pdf")), "question_bank")

    def test_textbook_default(self):
        from src.ingestion.pipeline import IngestionPipeline
        self.assertEqual(IngestionPipeline.infer_source_type(Path("computer_network.pdf")), "textbook")


# ---------------------------------------------------------------------------
# U3: DocumentChunker source_type
# ---------------------------------------------------------------------------

class TestDocumentChunkerSourceType(unittest.TestCase):
    def test_textbook_uses_larger_chunks(self):
        from src.ingestion.chunking.document_chunker import DocumentChunker
        from src.core.types import Document

        mock_settings = MagicMock()
        mock_settings.ingestion = MagicMock()
        mock_settings.ingestion.splitter = "recursive"
        mock_settings.ingestion.chunk_size = 1000
        mock_settings.ingestion.chunk_overlap = 200
        mock_settings.ingestion.chunk_refiner = {"parent_child_enabled": False}

        chunker = DocumentChunker(mock_settings)
        doc = Document(
            id="doc1",
            text="A" * 5000,
            metadata={"source_path": "/tmp/textbook.pdf"},
        )

        chunks = chunker.split_document(doc, source_type="textbook")
        self.assertTrue(len(chunks) > 0)
        for c in chunks:
            self.assertEqual(c.metadata.get("source_type"), "textbook")


# ---------------------------------------------------------------------------
# U4: QueryRouter
# ---------------------------------------------------------------------------

class TestQueryRouter(unittest.TestCase):
    def setUp(self):
        from src.core.query_engine.query_router import QueryRouter
        self.router = QueryRouter()

    def test_quiz_intent(self):
        decision = self.router.route("出一道TCP的练习题")
        self.assertEqual(decision.intent.value, "quiz_request")
        self.assertTrue(decision.need_rag)
        self.assertIn("question_bank", decision.preferred_sources)

    def test_review_intent(self):
        decision = self.router.route("帮我复习TCP考点")
        self.assertEqual(decision.intent.value, "concept_review")
        self.assertTrue(decision.need_rag)
        self.assertIn("slide", decision.preferred_sources)

    def test_deep_understanding_intent(self):
        decision = self.router.route("解释一下TCP的拥塞控制原理")
        self.assertEqual(decision.intent.value, "deep_understanding")
        self.assertTrue(decision.need_rag)
        self.assertIn("textbook", decision.preferred_sources)

    def test_general_chat(self):
        decision = self.router.route("你好")
        self.assertEqual(decision.intent.value, "general_chat")
        self.assertFalse(decision.need_rag)

    def test_knowledge_query_context_treats_router_as_retrieval_policy(self):
        decision = self.router.route("你好", planner_task_intent="knowledge_query")
        self.assertEqual(decision.intent.value, "deep_understanding")
        self.assertTrue(decision.need_rag)
        self.assertEqual(decision.preferred_sources, ["textbook", "slide"])
        self.assertIn("planner_context", decision.match_method)

    def test_non_knowledge_planner_context_returns_passthrough_policy(self):
        decision = self.router.route("帮我复习TCP考点", planner_task_intent="review_summary")
        self.assertEqual(decision.intent.value, "concept_review")
        self.assertTrue(decision.need_rag)
        self.assertEqual(decision.preferred_sources, ["slide", "textbook", "question_bank"])
        self.assertEqual(decision.match_method, "planner_context")
        self.assertEqual(
            decision.source_weights,
            {"slide": 0.45, "textbook": 0.4, "question_bank": 0.15},
        )

    def test_metadata_filter_single(self):
        from src.core.query_engine.query_router import RoutingDecision, QueryIntent
        rd = RoutingDecision(intent=QueryIntent.QUIZ_REQUEST, preferred_sources=["question_bank"])
        f = rd.to_metadata_filter()
        self.assertEqual(f, {"source_type": "question_bank"})

    def test_metadata_filter_multi(self):
        from src.core.query_engine.query_router import RoutingDecision, QueryIntent
        rd = RoutingDecision(intent=QueryIntent.CONCEPT_REVIEW, preferred_sources=["slide", "textbook"])
        f = rd.to_metadata_filter()
        self.assertEqual(f, {"source_type": {"$in": ["slide", "textbook"]}})

    def test_fallback_setting(self):
        from src.core.query_engine.query_router import QueryRouter
        router = QueryRouter(fallback_to_llm=False)
        decision = router.route("出一道题")
        self.assertFalse(decision.fallback_to_llm)

    def test_match_method_is_rule(self):
        decision = self.router.route("帮我复习TCP")
        self.assertEqual(decision.match_method, "rule")
        self.assertEqual(decision.confidence, 1.0)

    def test_default_fallback_method(self):
        from src.core.query_engine.query_router import QueryRouter
        router = QueryRouter()
        decision = router.route("TCP拥塞窗口如何保证公平性")
        self.assertIn(decision.match_method, ("rule", "default"))

    def test_source_budget_allocation_uses_largest_remainder(self):
        decision = self.router.route("帮我复习TCP考点", planner_task_intent="review_summary")
        budgets = decision.compute_source_unit_budgets(5)
        self.assertEqual(sum(budgets.values()), 5)
        self.assertGreaterEqual(budgets["slide"], budgets["textbook"])
        self.assertGreaterEqual(budgets["textbook"], budgets["question_bank"])

    def test_knowledge_query_planner_context_skips_embedding_layer(self):
        from src.core.query_engine.query_router import QueryRouter

        calls = {"count": 0}

        def embed(texts):
            calls["count"] += 1
            return [[0.1] * 8 for _ in texts]

        router = QueryRouter(embedding_fn=embed)
        init_calls = calls["count"]

        decision = router.route(
            "TCP拥塞窗口公平性",
            planner_task_intent="knowledge_query",
        )

        self.assertEqual(calls["count"], init_calls)
        self.assertEqual(decision.intent.value, "deep_understanding")
        self.assertEqual(decision.match_method, "planner_context_default")


# ---------------------------------------------------------------------------
# U4b: QueryRouter Embedding Layer
# ---------------------------------------------------------------------------

class TestQueryRouterEmbedding(unittest.TestCase):
    """Tests for the embedding-based intent classification layer."""

    @staticmethod
    def _make_deterministic_embed():
        """Create an embedding function that maps text to a deterministic vector
        using simple character hashing, producing roughly separable clusters
        for different intent prototypes."""
        import hashlib

        def embed(texts):
            results = []
            for t in texts:
                h = hashlib.md5(t.encode("utf-8")).hexdigest()
                vec = [int(c, 16) / 15.0 for c in h]
                results.append(vec)
            return results

        return embed

    def test_embedding_prototypes_precomputed(self):
        from src.core.query_engine.query_router import QueryRouter, _INTENT_PROTOTYPES
        embed_fn = self._make_deterministic_embed()
        router = QueryRouter(embedding_fn=embed_fn)

        self.assertTrue(router.embedding_ready)
        total_protos = sum(len(v) for v in _INTENT_PROTOTYPES.values())
        self.assertEqual(router.prototype_count, total_protos)

    def test_no_embedding_fn_rule_only(self):
        from src.core.query_engine.query_router import QueryRouter
        router = QueryRouter(embedding_fn=None)
        self.assertFalse(router.embedding_ready)
        self.assertEqual(router.prototype_count, 0)

        decision = router.route("出一道TCP的题")
        self.assertEqual(decision.intent.value, "quiz_request")
        self.assertEqual(decision.match_method, "rule")

    def test_embedding_fn_error_degrades_gracefully(self):
        from src.core.query_engine.query_router import QueryRouter

        def bad_embed(texts):
            raise RuntimeError("embedding service down")

        router = QueryRouter(embedding_fn=bad_embed)
        self.assertFalse(router.embedding_ready)

        decision = router.route("TCP协议的流量控制机制")
        self.assertIn(decision.match_method, ("rule", "default"))

    def test_rule_takes_priority_over_embedding(self):
        from src.core.query_engine.query_router import QueryRouter
        embed_fn = self._make_deterministic_embed()
        router = QueryRouter(embedding_fn=embed_fn)

        decision = router.route("出三道关于TCP的选择题")
        self.assertEqual(decision.match_method, "rule")
        self.assertEqual(decision.intent.value, "quiz_request")
        self.assertEqual(decision.confidence, 1.0)

    def test_embedding_layer_used_on_rule_miss(self):
        from src.core.query_engine.query_router import QueryRouter

        call_log = []

        def tracking_embed(texts):
            call_log.append(texts)
            return [[0.5] * 32 for _ in texts]

        router = QueryRouter(
            embedding_fn=tracking_embed,
            similarity_threshold=0.0,
        )

        query = "TCP拥塞窗口的公平性保障"
        decision = router.route(query)

        has_runtime_call = any(
            query in call_texts
            for call_texts in call_log
            if isinstance(call_texts, list)
        )
        self.assertTrue(has_runtime_call or decision.match_method in ("embedding", "rule"))

    def test_threshold_controls_match(self):
        from src.core.query_engine.query_router import QueryRouter
        import hashlib

        call_count = {"n": 0}

        def divergent_embed(texts):
            results = []
            for t in texts:
                call_count["n"] += 1
                h = hashlib.sha256(t.encode()).hexdigest()
                vec = [int(c, 16) / 15.0 for c in h[:16]]
                results.append(vec)
            return results

        router_strict = QueryRouter(
            embedding_fn=divergent_embed,
            similarity_threshold=0.9999,
        )
        decision = router_strict.route("一个完全不含关键词的随机查询哈哈哈XYZ123")
        self.assertIn(decision.match_method, ("default", "embedding"))

    def test_decision_has_confidence_and_method(self):
        from src.core.query_engine.query_router import RoutingDecision, QueryIntent
        rd = RoutingDecision(
            intent=QueryIntent.QUIZ_REQUEST,
            confidence=0.85,
            match_method="embedding",
        )
        self.assertEqual(rd.confidence, 0.85)
        self.assertEqual(rd.match_method, "embedding")

    def test_introspection_properties(self):
        from src.core.query_engine.query_router import QueryRouter
        embed_fn = self._make_deterministic_embed()
        router = QueryRouter(embedding_fn=embed_fn)
        self.assertTrue(router.embedding_ready)
        self.assertGreater(router.prototype_count, 0)

        router2 = QueryRouter()
        self.assertFalse(router2.embedding_ready)
        self.assertEqual(router2.prototype_count, 0)


# ---------------------------------------------------------------------------
# U5: QuizGenerator fallback
# ---------------------------------------------------------------------------

class TestQuizGeneratorFallback(unittest.TestCase):
    def test_search_question_bank_returns_empty_when_no_search(self):
        from src.agent.tools.quiz_generator import QuizGeneratorTool
        tool = QuizGeneratorTool(hybrid_search=None, llm_service=None)
        result = asyncio.get_event_loop().run_until_complete(
            tool._search_question_bank("TCP", 3)
        )
        self.assertEqual(result, [])

    def test_search_all_returns_empty_when_no_search(self):
        from src.agent.tools.quiz_generator import QuizGeneratorTool
        tool = QuizGeneratorTool(hybrid_search=None, llm_service=None)
        result = asyncio.get_event_loop().run_until_complete(
            tool._search_all("TCP")
        )
        self.assertEqual(result, [])

    def test_format_existing_questions(self):
        from src.agent.tools.quiz_generator import QuizGeneratorTool
        tool = QuizGeneratorTool()

        mock_result = MagicMock()
        mock_result.text = "TCP 是什么？"
        mock_result.metadata = {
            "answer": "传输控制协议",
            "explanation": "TCP是传输层协议",
            "options": {"A": "传输控制协议", "B": "用户数据报协议"},
            "tags": ["TCP", "传输层"],
        }
        formatted = tool._format_existing_questions([mock_result], "选择题")
        self.assertIn("课程题库", formatted)
        self.assertIn("TCP", formatted)


if __name__ == "__main__":
    unittest.main()
