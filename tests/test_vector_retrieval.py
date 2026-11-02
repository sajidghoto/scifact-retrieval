from app.vector_retrieval import aggregate_chunk_scores


def test_aggregate_uses_max_score_and_unique_deterministic_order():
    rows = [(1, "b", 0.8), (2, "a", 0.9), (3, "b", 0.95), (4, "c", 0.9)]
    assert aggregate_chunk_scores(rows, 3) == [
        ("b", 0.95),
        ("a", 0.9),
        ("c", 0.9),
    ]
