"""Pure rank-fusion functions for hybrid retrieval."""

from collections.abc import Sequence

RankedScores = Sequence[tuple[str, float]]


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero")


def reciprocal_rank_fusion(
    bm25: RankedScores, vector: RankedScores, *, k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked lists using reciprocal rank fusion."""
    _validate_k(k)
    scores: dict[str, float] = {}
    for ranked in (bm25, vector):
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _min_max_scores(ranked: RankedScores) -> dict[str, float]:
    if not ranked:
        return {}
    values = [score for _, score in ranked]
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return {doc_id: 1.0 for doc_id, _ in ranked}
    return {
        doc_id: (score - minimum) / (maximum - minimum)
        for doc_id, score in ranked
    }


def weighted_score_fusion(
    bm25: RankedScores, vector: RankedScores, *, alpha: float = 0.5
) -> list[tuple[str, float]]:
    """Fuse min-max-normalized scores; absent-list scores are zero."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if alpha == 0.0:
        present = {doc_id for doc_id, _ in bm25}
        return list(bm25) + [
            (doc_id, 0.0) for doc_id, _ in vector if doc_id not in present
        ]
    if alpha == 1.0:
        present = {doc_id for doc_id, _ in vector}
        return list(vector) + [
            (doc_id, 0.0) for doc_id, _ in bm25 if doc_id not in present
        ]
    bm25_scores = _min_max_scores(bm25)
    vector_scores = _min_max_scores(vector)
    doc_ids = set(bm25_scores) | set(vector_scores)
    scores = {
        doc_id: (1.0 - alpha) * bm25_scores.get(doc_id, 0.0)
        + alpha * vector_scores.get(doc_id, 0.0)
        for doc_id in doc_ids
    }
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
