import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "eval"))

from baseline import BM25, Document, evaluate, ndcg_at_k, recall_at_k


def test_bm25_ranks_matching_document_first() -> None:
    index = BM25(
        [
            Document("a", ["unrelated", "text"]),
            Document("b", ["cell", "biology"]),
        ]
    )
    assert index.search("cell biology", 2)[0][0] == "b"


def test_metrics_use_binary_relevance() -> None:
    ranked = ["wrong", "right", "other"]
    relevant = {"right"}
    assert recall_at_k(ranked, relevant, 2) == 1
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(1 / 1.5849625007)


def test_evaluate_requires_known_query_ids() -> None:
    index = BM25([Document("doc", ["term"])])
    with pytest.raises(KeyError):
        evaluate(index, {}, {"query": {"doc"}})
