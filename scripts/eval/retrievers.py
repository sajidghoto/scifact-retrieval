"""Retriever interfaces and local retriever implementations."""

from collections.abc import Iterable
from typing import Protocol

from app.retrieval import BM25, DocumentRecord


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return unique document IDs, ranked best first."""


class BM25Retriever:
    """Adapter exposing the existing BM25 implementation as a Retriever."""

    def __init__(self, documents: Iterable[DocumentRecord]) -> None:
        self.index = BM25(documents)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.index.search(query, k)
