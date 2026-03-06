"""Sparse Encoder for generating BM25 term statistics from text chunks.

This module implements the Sparse Encoder component of the Ingestion Pipeline,
responsible for extracting term statistics needed for BM25 indexing.

Design Principles:
- Stateless Processing: No internal state between encode() calls
- Observable: Accepts TraceContext for future observability integration
- Deterministic: Same inputs produce same term statistics
- Clear Contracts: Well-defined output structure for downstream BM25Indexer
- Chinese-Aware: Uses jieba for Chinese segmentation with English regex fallback
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from src.core.types import Chunk

logger = logging.getLogger(__name__)

try:
    import jieba

    jieba.setLogLevel(logging.WARNING)
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9][\w-]*")

_CHINESE_STOPWORDS: Set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "所以", "因为", "但是", "而且", "或者", "如果", "虽然", "然而",
    "可以", "能", "将", "把", "被", "让", "给", "对", "从", "以",
    "与", "及", "等", "之", "其", "中", "为", "而", "于", "并",
}


class SparseEncoder:
    """Encodes text chunks into BM25 term statistics.

    Uses jieba for Chinese text segmentation, and standard regex for
    English / numeric tokens.  Mixed-language text is handled seamlessly.
    """

    def __init__(
        self,
        min_term_length: int = 1,
        lowercase: bool = True,
        use_stopwords: bool = True,
    ):
        if min_term_length < 1:
            raise ValueError(f"min_term_length must be >= 1, got {min_term_length}")

        self.min_term_length = min_term_length
        self.lowercase = lowercase
        self._stopwords: Set[str] = _CHINESE_STOPWORDS.copy() if use_stopwords else set()
        if not _JIEBA_AVAILABLE:
            logger.warning("jieba not installed; Chinese segmentation disabled")
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Encode chunks into BM25 term statistics.
        
        For each chunk, extracts:
        - Term frequencies (term -> count)
        - Document length (total terms)
        - Unique terms count
        
        Args:
            chunks: List of Chunk objects to encode
            trace: Optional TraceContext for observability (reserved for Stage F)
        
        Returns:
            List of statistics dictionaries (one per chunk, in same order).
            Each dict contains: chunk_id, term_frequencies, doc_length, unique_terms
        
        Raises:
            ValueError: If chunks list is empty
            ValueError: If any chunk has empty text
        
        Example:
            >>> chunks = [
            ...     Chunk(id="1", text="machine learning", metadata={}),
            ...     Chunk(id="2", text="deep learning networks", metadata={})
            ... ]
            >>> stats = encoder.encode(chunks)
            >>> len(stats) == len(chunks)  # True
            >>> stats[0]["term_frequencies"]["machine"]  # 1
            >>> stats[1]["doc_length"]  # 3
        """
        if not chunks:
            raise ValueError("Cannot encode empty chunks list")
        
        results = []
        
        for i, chunk in enumerate(chunks):
            # Validate chunk text
            if not chunk.text or not chunk.text.strip():
                raise ValueError(
                    f"Chunk at index {i} (id={chunk.id}) has empty or whitespace-only text"
                )
            
            # Tokenize and count terms
            terms = self._tokenize(chunk.text)
            term_frequencies = Counter(terms)
            
            # Build statistics dict
            stat_dict = {
                "chunk_id": chunk.id,
                "term_frequencies": dict(term_frequencies),  # Convert Counter to dict
                "doc_length": len(terms),
                "unique_terms": len(term_frequencies),
            }
            
            results.append(stat_dict)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using jieba (Chinese) + regex (English).

        Mixed-language text is split by scanning for CJK ranges.
        Chinese segments are passed through ``jieba.cut``; non-Chinese
        segments are tokenised with a simple ``\\w+`` regex.
        """
        tokens: List[str] = []

        if _JIEBA_AVAILABLE and _CJK_RE.search(text):
            for word in jieba.cut(text):
                word = word.strip()
                if not word:
                    continue
                if self.lowercase:
                    word = word.lower()
                if len(word) < self.min_term_length:
                    continue
                if word in self._stopwords:
                    continue
                tokens.append(word)
        else:
            raw = _EN_TOKEN_RE.findall(text)
            if self.lowercase:
                raw = [t.lower() for t in raw]
            tokens = [t for t in raw if len(t) >= self.min_term_length and t not in self._stopwords]

        return tokens
    
    def get_corpus_stats(
        self,
        encoded_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate corpus-level statistics from encoded chunks.
        
        Utility method for BM25Indexer to compute:
        - Average document length
        - Document frequency (how many docs contain each term)
        - Total number of documents
        
        Args:
            encoded_chunks: List of statistics dicts from encode()
        
        Returns:
            Dictionary with corpus-level statistics:
            {
                "num_docs": int,
                "avg_doc_length": float,
                "document_frequency": Dict[str, int]  # term -> # docs containing it
            }
        """
        if not encoded_chunks:
            return {
                "num_docs": 0,
                "avg_doc_length": 0.0,
                "document_frequency": {}
            }
        
        num_docs = len(encoded_chunks)
        total_length = sum(chunk["doc_length"] for chunk in encoded_chunks)
        avg_doc_length = total_length / num_docs if num_docs > 0 else 0.0
        
        # Calculate document frequency (DF) for each term
        doc_freq: Dict[str, int] = {}
        for chunk_stats in encoded_chunks:
            # Each unique term in this chunk contributes 1 to DF
            for term in chunk_stats["term_frequencies"].keys():
                doc_freq[term] = doc_freq.get(term, 0) + 1
        
        return {
            "num_docs": num_docs,
            "avg_doc_length": avg_doc_length,
            "document_frequency": doc_freq,
        }
