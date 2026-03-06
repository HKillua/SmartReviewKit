"""Query Enhancement: Rewriting, HyDE, and Multi-Query decomposition.

Provides three complementary strategies to bridge the semantic gap between
user queries and indexed documents:

1. **Query Rewriting** — LLM reformulates the query for better retrieval.
2. **HyDE** — LLM generates a hypothetical answer; its embedding replaces
   the raw query embedding for dense retrieval.
3. **Multi-Query** — LLM decomposes a complex question into 2-4 sub-queries
   that are each retrieved independently and merged.

All three are optional and gated by ``settings.retrieval.*_enabled`` flags.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_REWRITE_PROMPT = (
    "你是一个信息检索专家。请将以下用户问题改写为更适合从知识库中检索的查询。\n"
    "要求：\n"
    "1. 保留原始语义\n"
    "2. 补充隐含的上下文关键词\n"
    "3. 去掉口语化表述\n"
    "4. 只输出改写后的查询，不要加任何解释\n\n"
    "用户问题：{query}\n"
    "改写后的查询："
)

_DEFAULT_HYDE_PROMPT = (
    "请你假设自己是一本计算机网络教材，针对以下问题写一段 100-200 字的回答。\n"
    "回答应包含相关专业术语和概念，便于向量检索匹配。\n"
    "只输出回答内容，不要加前缀。\n\n"
    "问题：{query}\n"
    "回答："
)

_DEFAULT_MULTI_QUERY_PROMPT = (
    "你是一个信息检索专家。请将以下复杂问题分解为 2-4 个独立的子查询，\n"
    "每个子查询聚焦一个具体方面，便于从知识库中分别检索。\n"
    "以 JSON 数组形式输出，例如 [\"子查询1\", \"子查询2\"]。\n"
    "只输出 JSON 数组，不要加任何解释。\n\n"
    "问题：{query}\n"
    "子查询："
)


def _load_prompt(path: str, fallback: str) -> str:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return fallback


class QueryEnhancer:
    """Unified entry-point for query enhancement strategies."""

    def __init__(
        self,
        llm_service: Any = None,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        rewrite_prompt_path: str = "config/prompts/query_rewrite.txt",
        hyde_prompt_path: str = "config/prompts/hyde.txt",
        multi_query_prompt_path: str = "config/prompts/multi_query.txt",
    ) -> None:
        self._llm = llm_service
        self._embed_fn = embedding_fn
        self._rewrite_prompt = _load_prompt(rewrite_prompt_path, _DEFAULT_REWRITE_PROMPT)
        self._hyde_prompt = _load_prompt(hyde_prompt_path, _DEFAULT_HYDE_PROMPT)
        self._multi_query_prompt = _load_prompt(multi_query_prompt_path, _DEFAULT_MULTI_QUERY_PROMPT)

    @property
    def llm_service(self) -> Any:
        return self._llm

    @llm_service.setter
    def llm_service(self, val: Any) -> None:
        self._llm = val

    # ------------------------------------------------------------------
    # Query Rewriting
    # ------------------------------------------------------------------

    async def rewrite(self, query: str) -> str:
        """Rewrite *query* using the LLM for better retrieval phrasing."""
        if self._llm is None:
            return query
        try:
            prompt = self._rewrite_prompt.replace("{query}", query)
            messages = [{"role": "user", "content": prompt}]
            response = await self._llm.chat(messages)
            rewritten = response.content.strip().strip('"').strip("'")
            if rewritten:
                logger.debug("Query rewritten: '%s' -> '%s'", query[:40], rewritten[:40])
                return rewritten
        except Exception as exc:
            logger.warning("Query rewrite failed, using original: %s", exc)
        return query

    # ------------------------------------------------------------------
    # HyDE  (Hypothetical Document Embedding)
    # ------------------------------------------------------------------

    async def hyde_embed(self, query: str) -> Optional[List[float]]:
        """Generate a hypothetical document for *query* and return its embedding.

        Returns ``None`` if either the LLM or embedding function is unavailable.
        """
        if self._llm is None or self._embed_fn is None:
            return None
        try:
            prompt = self._hyde_prompt.replace("{query}", query)
            messages = [{"role": "user", "content": prompt}]
            response = await self._llm.chat(messages)
            hypo_doc = response.content.strip()
            if not hypo_doc:
                return None
            vectors = self._embed_fn([hypo_doc])
            if vectors and len(vectors) > 0:
                logger.debug("HyDE doc generated (%d chars)", len(hypo_doc))
                return vectors[0]
        except Exception as exc:
            logger.warning("HyDE embedding failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Multi-Query Decomposition
    # ------------------------------------------------------------------

    async def decompose(self, query: str) -> List[str]:
        """Decompose *query* into 2-4 independent sub-queries."""
        if self._llm is None:
            return [query]
        try:
            prompt = self._multi_query_prompt.replace("{query}", query)
            messages = [{"role": "user", "content": prompt}]
            response = await self._llm.chat(messages)
            text = response.content.strip()
            if text.startswith("```"):
                text = text.strip("`").strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            sub_queries = json.loads(text)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                logger.debug("Query decomposed into %d sub-queries", len(sub_queries))
                return sub_queries[:4]
        except Exception as exc:
            logger.warning("Multi-query decomposition failed: %s", exc)
        return [query]
