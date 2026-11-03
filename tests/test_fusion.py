import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.fusion import reciprocal_rank_fusion, weighted_score_fusion


def test_rrf_handles_one_list_and_ties_deterministically():
    assert reciprocal_rank_fusion([("b", 1.0)], [("a", 1.0)], k=10) == [
        ("a", pytest.approx(1 / 11)),
        ("b", pytest.approx(1 / 11)),
    ]


def test_rrf_identical_lists_and_empty_list():
    assert [doc for doc, _ in reciprocal_rank_fusion([("a", 2.0), ("b", 1.0)], [("a", 1.0), ("b", 0.0)], k=1)] == ["a", "b"]
    assert reciprocal_rank_fusion([], [], k=60) == []


def test_weighted_missing_documents_and_extremes():
    bm25 = [("b", 2.0), ("a", 1.0)]
    vector = [("a", 0.9), ("c", 0.1)]
    assert [doc for doc, _ in weighted_score_fusion(bm25, vector, alpha=0)] == ["b", "a", "c"]
    assert [doc for doc, _ in weighted_score_fusion(bm25, vector, alpha=1)] == ["a", "c", "b"]


def test_weighted_ties_empty_and_validation():
    assert weighted_score_fusion([("b", 1.0), ("a", 1.0)], []) == [
        ("a", 0.5),
        ("b", 0.5),
    ]
    assert weighted_score_fusion([], []) == []
    with pytest.raises(ValueError):
        weighted_score_fusion([], [], alpha=1.1)
