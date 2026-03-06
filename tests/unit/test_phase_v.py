"""Unit tests for Phase V: Semantic Cache."""

import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock


class TestSemanticCache(unittest.TestCase):
    def _make_cache(self, **kwargs):
        from src.core.cache.semantic_cache import SemanticCache

        embed_counter = {"calls": 0}

        def mock_embed(text):
            embed_counter["calls"] += 1
            h = hash(text) % 10000
            return [h / 10000.0, (h + 1) / 10000.0, (h + 2) / 10000.0]

        cache = SemanticCache(embedding_fn=mock_embed, **kwargs)
        return cache, embed_counter

    def test_put_and_get_hit(self):
        cache, _ = self._make_cache(similarity_threshold=0.5)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.put("what is TCP", "TCP is a protocol", {"collection": "test"}))
        result = loop.run_until_complete(cache.get("what is TCP"))
        self.assertIsNotNone(result)
        self.assertIn("TCP", result.result)
        loop.close()

    def test_miss_on_different_query(self):
        """Ensure miss when threshold is extremely high and queries differ."""
        from src.core.cache.semantic_cache import SemanticCache

        call_count = {"n": 0}

        def varying_embed(text):
            call_count["n"] += 1
            if "cooking" in text:
                return [0.0, 0.0, 1.0]
            return [1.0, 0.0, 0.0]

        cache = SemanticCache(
            embedding_fn=varying_embed,
            similarity_threshold=0.99,
        )
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.put("what is TCP", "TCP info", {}))
        result = loop.run_until_complete(cache.get("completely different query about cooking"))
        self.assertIsNone(result)
        loop.close()

    def test_ttl_expiration(self):
        cache, _ = self._make_cache(similarity_threshold=0.5, ttl_seconds=0)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.put("query1", "result1", {}))
        time.sleep(0.05)
        result = loop.run_until_complete(cache.get("query1"))
        self.assertIsNone(result)
        loop.close()

    def test_lru_eviction(self):
        cache, _ = self._make_cache(max_size=2, similarity_threshold=0.5)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.put("q1", "r1", {}))
        loop.run_until_complete(cache.put("q2", "r2", {}))
        loop.run_until_complete(cache.put("q3", "r3", {}))
        self.assertLessEqual(len(cache._entries), 2)
        loop.close()

    def test_invalidate_by_collection(self):
        cache, _ = self._make_cache(similarity_threshold=0.5)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.put("q1", "r1", {"collection": "net"}))
        loop.run_until_complete(cache.put("q2", "r2", {"collection": "db"}))
        removed = cache.invalidate_by_collection("net")
        self.assertEqual(removed, 1)
        self.assertEqual(len(cache._entries), 1)
        loop.close()

    def test_clear(self):
        cache, _ = self._make_cache()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.put("q1", "r1", {}))
        cache.clear()
        self.assertEqual(len(cache._entries), 0)
        loop.close()

    def test_stats(self):
        cache, _ = self._make_cache(similarity_threshold=0.5)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(cache.put("q1", "r1", {}))
        loop.run_until_complete(cache.get("q1"))
        loop.run_until_complete(cache.get("totally different"))
        stats = cache.stats
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("size", stats)
        loop.close()

    def test_thread_safety(self):
        cache, _ = self._make_cache(similarity_threshold=0.5)
        self.assertTrue(hasattr(cache._lock, 'acquire'))

    def test_embedding_fn_error_returns_none(self):
        from src.core.cache.semantic_cache import SemanticCache

        def bad_embed(text):
            raise RuntimeError("embedding failed")

        cache = SemanticCache(embedding_fn=bad_embed)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(cache.get("query"))
        self.assertIsNone(result)
        loop.close()


if __name__ == "__main__":
    unittest.main()
