"""End-to-end retrieval tests against the deterministic in-memory backend."""

from clipopedia.config import get_settings
from clipopedia.factory import build_demo_backend


async def test_pipeline_finds_relevant_clip():
    backend = await build_demo_backend(get_settings(refresh=True))
    _, selection = await backend.pipeline.run("best clip on AI agents and reliability")
    assert selection is not None
    blob = (selection.chunk.clip.text + " " + " ".join(selection.chunk.clip.topics)).lower()
    assert "agent" in blob


def test_pipeline_burnout_query():
    import asyncio

    async def _run():
        backend = await build_demo_backend(get_settings(refresh=True))
        return await backend.pipeline.run("founder burnout advice")

    _, selection = asyncio.run(_run())
    assert selection is not None
    blob = (selection.chunk.clip.text + " " + " ".join(selection.chunk.clip.topics)).lower()
    assert "burnout" in blob


async def test_small_talk_short_circuits():
    backend = await build_demo_backend(get_settings(refresh=True))
    analysis, selection = await backend.pipeline.run("hi")
    assert analysis.is_small_talk is True
    assert selection is None


async def test_pipeline_returns_within_final_top_k_ranking():
    backend = await build_demo_backend(get_settings(refresh=True))
    analysis = await backend.pipeline.analyze("startup pricing strategy")
    chunks = await backend.pipeline.retrieve("startup pricing strategy", analysis)
    assert chunks  # non-empty
    # final_score must be monotonically non-increasing after ranking.
    scores = [c.final_score for c in chunks]
    assert scores == sorted(scores, reverse=True)
