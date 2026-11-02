"""Retriever-independent ranking metrics."""

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(
    ranked: Sequence[str], relevant: set[str], k: int
) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    ideal_length = min(k, len(relevant))
    if not ideal_length:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal_dcg = sum(
        1 / math.log2(rank + 1) for rank in range(1, ideal_length + 1)
    )
    return dcg / ideal_dcg


def evaluate_rankings(
    rankings: dict[str, Sequence[str]], qrels: dict[str, set[str]]
) -> dict[str, float | int]:
    if not qrels:
        raise ValueError("At least one qrel is required")
    missing = sorted(set(qrels) - set(rankings))
    if missing:
        raise KeyError(missing[0])
    return {
        "queries": len(qrels),
        "relevant_pairs": sum(len(relevant) for relevant in qrels.values()),
        "recall@10": sum(
            recall_at_k(rankings[query_id], relevant, 10)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
        "recall@100": sum(
            recall_at_k(rankings[query_id], relevant, 100)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
        "mrr@10": sum(
            reciprocal_rank_at_k(rankings[query_id], relevant, 10)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
        "ndcg@10": sum(
            ndcg_at_k(rankings[query_id], relevant, 10)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
    }
