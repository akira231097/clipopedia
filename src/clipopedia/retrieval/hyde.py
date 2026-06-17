"""HyDE (Hypothetical Document Embeddings) weighting.

Rather than embedding only the user's terse query, the analyzer asks an LLM to
write a few *hypothetical* answer snippets — the kind of transcript passage that
would perfectly answer the question. Those richer texts embed closer to real
relevant chunks. We search with the original query **and** each hypothetical
document, then fuse.

Not every hypothetical is equally good, though. This module scores each one by
cosine similarity to the original query embedding and assigns a fusion weight
that decays from ``hyde_high`` (most on-topic) to ``hyde_low`` (most divergent).
The original query always keeps the largest weight.

Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
Labels" (2022).
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def compute_query_weights(
    vectors: list[list[float]],
    *,
    original_weight: float,
    hyde_high: float,
    hyde_low: float,
) -> list[float]:
    """Return one fusion weight per query vector.

    ``vectors[0]`` is the original query; the rest are HyDE documents. The
    returned list is aligned to ``vectors`` so it can be passed straight into
    :func:`reciprocal_rank_fusion`.
    """
    n = len(vectors)
    if n == 0:
        return []
    weights = [original_weight] + [0.0] * (n - 1)
    if n == 1:
        return weights

    if hyde_high < hyde_low:
        hyde_high, hyde_low = hyde_low, hyde_high

    base = vectors[0]
    sims = [(i, cosine_similarity(base, vectors[i])) for i in range(1, n)]
    # Highest similarity → highest weight; interpolate linearly down to the min.
    sims.sort(key=lambda t: t[1], reverse=True)
    m = len(sims)
    step = (hyde_high - hyde_low) / max(m - 1, 1) if m > 1 else 0.0
    for position, (idx, _sim) in enumerate(sims):
        weights[idx] = round(max(hyde_low, hyde_high - step * position), 4)
    return weights
