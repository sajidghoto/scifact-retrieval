"""Exact pgvector retrieval over stored chunk embeddings."""

from collections.abc import Sequence
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class QueryEmbedder(Protocol):
    def embed(self, text: str) -> Sequence[float]:
        ...


def aggregate_chunk_scores(
    rows: Sequence[tuple[int, str, float]], limit: int
) -> list[tuple[str, float]]:
    """Keep the maximum chunk similarity per document and rank deterministically."""
    best: dict[str, float] = {}
    for _, doc_id, score in rows:
        best[doc_id] = max(score, best.get(doc_id, float("-inf")))
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))[:limit]


class VectorRetriever:
    def __init__(
        self,
        session: Session,
        embedder: QueryEmbedder,
        *,
        fetch_initial: int = 1000,
        max_fetch: int | None = None,
        use_query_prefix: bool = True,
    ) -> None:
        if fetch_initial <= 0:
            raise ValueError("fetch_initial must be greater than zero")
        self.session = session
        self.embedder = embedder
        self.fetch_initial = fetch_initial
        self.max_fetch = max_fetch
        self.use_query_prefix = use_query_prefix
        self.last_fetch_count = 0
        self.last_distinct_documents = 0

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        text = f"{QUERY_PREFIX}{query}" if self.use_query_prefix else query
        vector = list(self.embedder.embed(text))
        return self.search_vector(vector, k)

    def search_vector(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be greater than zero")
        fetch_count = self.fetch_initial
        max_fetch = self.max_fetch or self.session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
        )
        max_fetch = max(fetch_count, int(max_fetch or fetch_count))
        rows: list[tuple[int, str, float]] = []
        while True:
            distance = Chunk.embedding.cosine_distance(vector)
            rows = [
                (int(chunk_id), doc_id, 1.0 - float(distance))
                for chunk_id, doc_id, distance in self.session.execute(
                    select(Chunk.id, Chunk.doc_id, distance)
                    .where(Chunk.embedding.is_not(None))
                    .order_by(distance, Chunk.id)
                    .limit(fetch_count)
                )
            ]
            distinct = len({doc_id for _, doc_id, _ in rows})
            if distinct >= 100 or fetch_count >= max_fetch:
                self.last_fetch_count = fetch_count
                self.last_distinct_documents = distinct
                return aggregate_chunk_scores(rows, k)
            fetch_count = min(fetch_count * 2, max_fetch)
