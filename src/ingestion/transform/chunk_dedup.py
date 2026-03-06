"""Chunk-level semantic deduplication using SimHash with LSH bucketing.

SimHash produces a fixed-width fingerprint per chunk such that
similar texts yield similar hashes.  Two hashes are considered
duplicates if their Hamming distance is below a configurable
threshold (default: 3 bits out of 64).

Performance: Uses Locality-Sensitive Hashing (band partitioning)
to avoid the naive O(n^2) pairwise comparison.  The 64-bit hash
is split into ``num_bands`` bands; candidates that share at least
one band are compared exactly.  Average complexity is near O(n).
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from typing import List

from src.core.types import Chunk

logger = logging.getLogger(__name__)

try:
    import jieba
    _JIEBA = True
except ImportError:
    _JIEBA = False

import re

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


def _band_keys(fingerprint: int, num_bands: int = 4) -> List[int]:
    """Split a 64-bit fingerprint into ``num_bands`` sub-hashes for LSH."""
    band_width = HASH_BITS // num_bands
    keys: List[int] = []
    for b in range(num_bands):
        shift = b * band_width
        mask = (1 << band_width) - 1
        keys.append((fingerprint >> shift) & mask)
    return keys


def dedup_chunks(
    chunks: List[Chunk],
    threshold: int = 3,
    num_bands: int = 4,
) -> List[Chunk]:
    """Remove near-duplicate chunks based on SimHash Hamming distance.

    Uses LSH band partitioning to reduce average complexity from O(n^2)
    to approximately O(n).  Chunks that share at least one band key
    are compared exactly via Hamming distance.

    Args:
        chunks: Input list of chunks.
        threshold: Maximum Hamming distance to consider as duplicate.
        num_bands: Number of LSH bands (more bands = higher recall,
            slightly more comparisons).

    Returns:
        De-duplicated list (order preserved, first occurrence kept).
    """
    if threshold < 0 or threshold > HASH_BITS:
        threshold = max(0, min(threshold, HASH_BITS))

    # band_index -> list of (fingerprint, chunk_position)
    buckets: List[dict[int, List[int]]] = [
        defaultdict(list) for _ in range(num_bands)
    ]
    fingerprints: List[int] = []
    kept_indices: List[int] = []
    removed = 0

    for idx, chunk in enumerate(chunks):
        fp = simhash(chunk.text)
        fingerprints.append(fp)
        band_keys = _band_keys(fp, num_bands)

        # Collect candidate set from all bands
        candidate_set: set[int] = set()
        for band_idx, key in enumerate(band_keys):
            for prev_idx in buckets[band_idx][key]:
                candidate_set.add(prev_idx)

        # Check exact Hamming distance against candidates only
        is_dup = False
        for cand_idx in candidate_set:
            if hamming_distance(fp, fingerprints[cand_idx]) <= threshold:
                is_dup = True
                break

        if not is_dup:
            kept_indices.append(idx)
            for band_idx, key in enumerate(band_keys):
                buckets[band_idx][key].append(idx)
        else:
            removed += 1

    if removed:
        logger.info("SimHash dedup removed %d / %d chunks", removed, len(chunks))
    return [chunks[i] for i in kept_indices]
