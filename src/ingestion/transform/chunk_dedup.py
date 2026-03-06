"""Chunk-level semantic deduplication using SimHash.

SimHash produces a fixed-width fingerprint per chunk such that
similar texts yield similar hashes.  Two hashes are considered
duplicates if their Hamming distance is below a configurable
threshold (default: 3 bits out of 64).
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from typing import List, Set

from src.core.types import Chunk

logger = logging.getLogger(__name__)

try:
    import jieba
    _JIEBA = True
except ImportError:
    _JIEBA = False

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

HASH_BITS = 64


def _tokenize(text: str) -> List[str]:
    if _JIEBA and _CJK_RE.search(text):
        return [w for w in jieba.cut(text) if w.strip()]
    return _WORD_RE.findall(text)


def _token_hash(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def simhash(text: str) -> int:
    """Compute a 64-bit SimHash fingerprint for *text*."""
    tokens = _tokenize(text.lower())
    tf = Counter(tokens)
    v = [0] * HASH_BITS
    for token, weight in tf.items():
        h = _token_hash(token)
        for i in range(HASH_BITS):
            if h & (1 << i):
                v[i] += weight
            else:
                v[i] -= weight
    fingerprint = 0
    for i in range(HASH_BITS):
        if v[i] > 0:
            fingerprint |= 1 << i
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_chunks(chunks: List[Chunk], threshold: int = 3) -> List[Chunk]:
    """Remove near-duplicate chunks based on SimHash Hamming distance.

    Args:
        chunks: Input list of chunks.
        threshold: Maximum Hamming distance to consider as duplicate.

    Returns:
        De-duplicated list (order preserved, first occurrence kept).
    """
    seen_hashes: List[int] = []
    kept: List[Chunk] = []
    removed = 0

    for chunk in chunks:
        fp = simhash(chunk.text)
        is_dup = False
        for existing in seen_hashes:
            if hamming_distance(fp, existing) <= threshold:
                is_dup = True
                break
        if not is_dup:
            seen_hashes.append(fp)
            kept.append(chunk)
        else:
            removed += 1

    if removed:
        logger.info("SimHash dedup removed %d / %d chunks", removed, len(chunks))
    return kept
