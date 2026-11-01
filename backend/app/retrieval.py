"""Deterministic in-memory BM25 retrieval for indexed documents."""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
K1 = 1.2
B = 0.75


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    title: str
    text: str

    @property
    def tokens(self) -> list[str]:
        return tokenize(f"{self.title} {self.text}")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize alphanumeric terms."""
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


class BM25:
    """In-memory BM25 index with deterministic document-id tie breaking."""

    def __init__(
        self, documents: Iterable[DocumentRecord], k1: float = K1, b: float = B
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("At least one document is required")
        if k1 <= 0 or b < 0 or b > 1:
            raise ValueError("BM25 parameters must satisfy k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.lengths = [len(document.tokens) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.documents)
        if self.average_length == 0:
            raise ValueError("At least one document must contain a token")
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, document in enumerate(self.documents):
            for term, frequency in Counter(document.tokens).items():
                self.postings[term].append((index, frequency))

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        scores: defaultdict[int, float] = defaultdict(float)
        query_terms = Counter(tokenize(query))
        document_count = len(self.documents)
        for term, query_frequency in query_terms.items():
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = math.log(
                1 + (document_count - len(postings) + 0.5) / (len(postings) + 0.5)
            )
            for document_index, term_frequency in postings:
                length_ratio = self.lengths[document_index] / self.average_length
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length_ratio
                )
                scores[document_index] += (
                    idf
                    * (term_frequency * (self.k1 + 1) / denominator)
                    * query_frequency
                )
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self.documents[item[0]].doc_id),
        )
        return [
            (self.documents[index].doc_id, score) for index, score in ranked[:limit]
        ]
