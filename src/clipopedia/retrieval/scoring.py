"""Post-retrieval scoring: diversity caps, recency boosting, metadata signals.

The reranker gives us a good relevance ordering, but a production recommender
needs more than raw relevance:

* **Episode diversity** — don't return five near-identical chunks from the same
  episode; cap how many a single episode can contribute.
* **Recency** — for a discovery bot, a slightly-less-relevant clip from last
  week often beats a perfect match from three years ago.
* **Metadata agreement** — if the user named a guest/host/show and a chunk
  matches it, that is strong corroborating evidence.

Each adjustment is multiplicative and explainable, so the final ordering can be
traced back to concrete signals.
"""

from __future__ import annotations

from datetime import timedelta

from ..config import Settings
from ..dateutils import to_numeric, today_utc
from ..models import QueryAnalysis, RetrievedChunk, VectorMatch


def enforce_episode_cap_and_bucket_quota(
    matches: list[VectorMatch],
    *,
    per_episode_cap: int,
    min_per_bucket: int,
) -> list[VectorMatch]:
    """Cap per-episode contributions while guaranteeing bucket representation.

    Expects ``metadata['episode_id']`` and ``metadata['bucket']`` to be set
    (the pipeline attaches the bucket label during fusion).
    """
    by_episode: dict[str, int] = {}
    by_bucket: dict[str, int] = {}
    selected: list[VectorMatch] = []
    deferred: list[VectorMatch] = []

    for m in matches:
        epi = m.metadata.get("episode_id")
        if epi is not None and by_episode.get(epi, 0) >= per_episode_cap:
            deferred.append(m)
            continue
        if epi is not None:
            by_episode[epi] = by_episode.get(epi, 0) + 1
        selected.append(m)
        bucket = m.metadata.get("bucket", "all")
        by_bucket[bucket] = by_bucket.get(bucket, 0) + 1

    # Guarantee a minimum number of items per bucket by pulling back deferred
    # (capped) items from under-represented buckets.
    if min_per_bucket > 0 and deferred:
        for bucket, count in list(by_bucket.items()):
            need = min_per_bucket - count
            if need <= 0:
                continue
            for m in list(deferred):
                if need <= 0:
                    break
                if m.metadata.get("bucket", "all") == bucket:
                    selected.append(m)
                    deferred.remove(m)
                    need -= 1
    return selected


def _base_score(chunk: RetrievedChunk) -> float:
    """The relevance signal to build on: reranker score if present, else fused."""
    return chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score


def apply_metadata_boost(
    chunks: list[RetrievedChunk],
    analysis: QueryAnalysis,
    *,
    guest_boost: float = 1.15,
    host_boost: float = 1.10,
    show_boost: float = 1.10,
) -> None:
    """Boost chunks whose metadata agrees with named entities in the query."""
    wanted_guests = {g.lower() for g in analysis.guests}
    wanted_hosts = {h.lower() for h in analysis.hosts}
    wanted_show = (analysis.show or "").lower()

    for c in chunks:
        if wanted_guests and wanted_guests & {g.lower() for g in c.clip.guests}:
            c.final_score *= guest_boost
        if wanted_hosts and wanted_hosts & {h.lower() for h in c.clip.hosts}:
            c.final_score *= host_boost
        if wanted_show and wanted_show == c.clip.show_title.lower():
            c.final_score *= show_boost


def apply_recency_boost(chunks: list[RetrievedChunk], settings: Settings) -> None:
    """Multiply the score of chunks published within the recent window."""
    cutoff = to_numeric(today_utc() - timedelta(days=settings.recent_window_days))
    for c in chunks:
        pd = c.clip.published_date
        if pd and to_numeric(pd) >= cutoff:
            c.final_score *= settings.recency_boost


def apply_relevance_floor(
    chunks: list[RetrievedChunk], floor: float, *, min_keep: int = 3
) -> list[RetrievedChunk]:
    """Drop low-relevance chunks, but never return fewer than ``min_keep``."""
    above = [c for c in chunks if _base_score(c) >= floor]
    if len(above) >= min_keep:
        return above
    return chunks[:min_keep]


def score_and_rank(
    chunks: list[RetrievedChunk],
    analysis: QueryAnalysis,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Compute final scores from all signals and return chunks best-first."""
    for c in chunks:
        c.final_score = _base_score(c)
    apply_metadata_boost(chunks, analysis)
    apply_recency_boost(chunks, settings)
    chunks.sort(key=lambda c: c.final_score, reverse=True)
    return chunks
