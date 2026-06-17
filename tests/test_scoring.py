from datetime import timedelta

from clipopedia.config import Settings
from clipopedia.dateutils import today_utc
from clipopedia.models import Clip, QueryAnalysis, RetrievedChunk, VectorMatch
from clipopedia.retrieval.scoring import (
    apply_relevance_floor,
    enforce_episode_cap_and_bucket_quota,
    score_and_rank,
)

SETTINGS = Settings()


def _match(cid, episode, bucket="all"):
    return VectorMatch(chunk_id=cid, score=1.0, metadata={"episode_id": episode, "bucket": bucket})


def _chunk(cid, *, rerank=0.5, guests=None, days_ago=400):
    clip = Clip(
        chunk_id=cid,
        episode_id="e",
        text="x",
        guests=guests or [],
        published_date=today_utc() - timedelta(days=days_ago),
    )
    return RetrievedChunk(clip=clip, hybrid_score=0.1, rerank_score=rerank)


def test_episode_cap_limits_per_episode():
    matches = [_match(f"c{i}", "ep1") for i in range(5)] + [_match("c5", "ep2")]
    kept = enforce_episode_cap_and_bucket_quota(matches, per_episode_cap=2, min_per_bucket=0)
    ep1 = [m for m in kept if m.metadata["episode_id"] == "ep1"]
    assert len(ep1) == 2
    assert any(m.metadata["episode_id"] == "ep2" for m in kept)


def test_relevance_floor_keeps_minimum():
    chunks = [_chunk(f"c{i}", rerank=0.1) for i in range(5)]
    kept = apply_relevance_floor(chunks, floor=0.35, min_keep=3)
    assert len(kept) == 3  # all below floor, but min_keep honoured


def test_metadata_boost_promotes_entity_match():
    plain = _chunk("plain", rerank=0.6, guests=["Nobody"])
    matched = _chunk("matched", rerank=0.55, guests=["Dr. Lena Ortiz"])
    analysis = QueryAnalysis(cleaned_query="q", guests=["Dr. Lena Ortiz"])
    ranked = score_and_rank([plain, matched], analysis, SETTINGS)
    assert ranked[0].clip.chunk_id == "matched"  # boost overtakes higher base


def test_recency_boost_promotes_recent():
    old = _chunk("old", rerank=0.6, days_ago=400)
    new = _chunk("new", rerank=0.55, days_ago=5)
    ranked = score_and_rank([old, new], QueryAnalysis(cleaned_query="q"), SETTINGS)
    assert ranked[0].clip.chunk_id == "new"
