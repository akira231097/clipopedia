from clipopedia.retrieval.hyde import compute_query_weights, cosine_similarity


def test_cosine_basic():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([], [1, 0]) == 0.0


def test_original_query_keeps_top_weight():
    vecs = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
    weights = compute_query_weights(vecs, original_weight=1.25, hyde_high=1.1, hyde_low=0.85)
    assert weights[0] == 1.25
    assert len(weights) == 3


def test_more_similar_hyde_gets_higher_weight():
    base = [1.0, 0.0]
    similar = [0.95, 0.05]   # close to base
    divergent = [0.0, 1.0]   # orthogonal
    vecs = [base, similar, divergent]
    weights = compute_query_weights(vecs, original_weight=1.25, hyde_high=1.1, hyde_low=0.85)
    assert weights[1] > weights[2]
    assert weights[1] <= 1.1 and weights[2] >= 0.85


def test_single_query_no_hyde():
    weights = compute_query_weights([[1.0, 0.0]], original_weight=1.25, hyde_high=1.1, hyde_low=0.85)
    assert weights == [1.25]
