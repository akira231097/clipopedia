"""Weighted Reciprocal Rank Fusion (RRF).

RRF combines several independently-ranked result lists into one consensus
ranking using only each item's *rank* (not its raw score), which makes it
robust to score distributions that differ wildly between dense, sparse, and
per-HyDE-query searches. Each list can carry its own weight so that, for
example, the original query contributes more than a speculative HyDE document.

    score(item) = Σ_lists  weight_list / (k + rank_in_list)

See: Cormack et al., "Reciprocal Rank Fusion Outperforms Condorcet and
Individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..models import VectorMatch


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[VectorMatch]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
    top_k: int | None = None,
) -> list[VectorMatch]:
    """Fuse ranked lists of :class:`VectorMatch` into one ranking.

    Each input list must already be ordered best-first. Metadata for a given
    chunk is taken from the occurrence with the highest original score.
    """
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights length must match number of result lists")

    fused: dict[str, float] = {}
    best_seen: dict[str, VectorMatch] = {}

    for matches, weight in zip(result_lists, weights, strict=True):
        for rank, match in enumerate(matches):
            cid = match.chunk_id
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
            prev = best_seen.get(cid)
            if prev is None or match.score > prev.score:
                best_seen[cid] = match

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        ordered = ordered[:top_k]

    out: list[VectorMatch] = []
    for cid, fused_score in ordered:
        base = best_seen[cid]
        out.append(
            VectorMatch(chunk_id=cid, score=fused_score, metadata=base.metadata)
        )
    return out
