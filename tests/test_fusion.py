from clipopedia.models import VectorMatch
from clipopedia.retrieval.fusion import reciprocal_rank_fusion


def _matches(ids):
    return [VectorMatch(chunk_id=i, score=1.0) for i in ids]


def test_consensus_item_ranks_first():
    list_a = _matches(["x", "a", "b"])
    list_b = _matches(["x", "b", "c"])
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0].chunk_id == "x"  # top of both lists


def test_weights_shift_ranking():
    list_a = _matches(["a", "b"])
    list_b = _matches(["b", "a"])
    # Heavily favour the first list, which ranks "a" first.
    fused = reciprocal_rank_fusion([list_a, list_b], weights=[5.0, 1.0])
    assert fused[0].chunk_id == "a"


def test_top_k_truncates():
    fused = reciprocal_rank_fusion([_matches(["a", "b", "c", "d"])], top_k=2)
    assert len(fused) == 2


def test_metadata_from_best_score_occurrence():
    a = [VectorMatch(chunk_id="z", score=0.2, metadata={"src": "low"})]
    b = [VectorMatch(chunk_id="z", score=0.9, metadata={"src": "high"})]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].metadata["src"] == "high"


def test_mismatched_weights_raise():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion([_matches(["a"])], weights=[1.0, 2.0])
