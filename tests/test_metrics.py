import pytest
import numpy as np

from scripts.eval.bootstrap import paired_bootstrap
from scripts.eval.metrics import (
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_metrics_handle_miss_and_ties():
    assert recall_at_k(["a", "b"], {"z"}, 2) == 0
    assert reciprocal_rank_at_k(["a", "b"], {"z"}, 2) == 0
    assert ndcg_at_k(["a", "b"], {"z"}, 2) == 0
    assert recall_at_k(["a", "b"], {"b"}, 2) == 1
    assert reciprocal_rank_at_k(["a", "b"], {"b"}, 2) == 0.5


def test_ndcg_is_one_when_all_relevant_items_are_tied_in_top_k():
    assert ndcg_at_k(["b", "a", "c"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_evaluate_rankings_returns_macro_metrics():
    result = evaluate_rankings({"q1": ["d1"], "q2": ["x"]}, {"q1": {"d1"}, "q2": {"d2"}})
    assert result["queries"] == 2
    assert result["recall@10"] == 0.5


def test_paired_bootstrap_matches_hand_checkable_difference():
    first = {"ndcg@10": np.array([1.0, 0.0]), "recall@10": np.array([1.0, 0.0]), "mrr@10": np.array([1.0, 0.0])}
    second = {"ndcg@10": np.array([0.0, 0.0]), "recall@10": np.array([0.0, 0.0]), "mrr@10": np.array([0.0, 0.0])}

    result = paired_bootstrap(first, second, resamples=100, seed=7)

    for metric in ("ndcg@10", "recall@10", "mrr@10"):
        assert result[metric]["mean_difference"] == pytest.approx(0.5)
        assert result[metric]["ci95_percentile"] == pytest.approx([0.0, 1.0])
