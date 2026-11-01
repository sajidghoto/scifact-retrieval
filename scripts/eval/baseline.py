"""Run a deterministic BM25 retrieval baseline and evaluate SciFact qrels."""

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
K1 = 1.2
B = 0.75


@dataclass(frozen=True)
class Document:
    doc_id: str
    tokens: list[str]


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize alphanumeric terms."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def read_documents(path: Path) -> list[Document]:
    documents = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                doc_id = record["_id"]
                title = record.get("title", "")
                text = record["text"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid document at line {line_number}") from error
            if not isinstance(doc_id, str) or not isinstance(title, str):
                raise ValueError(f"Invalid document fields at line {line_number}")
            if not isinstance(text, str):
                raise ValueError(f"Invalid document text at line {line_number}")
            documents.append(Document(doc_id, tokenize(f"{title} {text}")))
    return documents


def read_queries(path: Path) -> dict[str, str]:
    queries = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                record = json.loads(line)
                query_id = record["_id"]
                text = record["text"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid query at line {line_number}") from error
            if not isinstance(query_id, str) or not isinstance(text, str):
                raise ValueError(f"Invalid query fields at line {line_number}")
            queries[query_id] = text
    return queries


def read_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Expected TSV columns {sorted(required)} in {path}")
        for record in reader:
            if record["score"] == "1":
                qrels[record["query-id"]].add(record["corpus-id"])
    return dict(qrels)


class BM25:
    """In-memory BM25 index with deterministic document-id tie breaking."""

    def __init__(
        self, documents: Iterable[Document], k1: float = K1, b: float = B
    ) -> None:
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("At least one document is required")
        if k1 <= 0 or b < 0 or b > 1:
            raise ValueError("BM25 parameters must satisfy k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self.lengths = [len(document.tokens) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
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


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    ideal_length = min(k, len(relevant))
    if not ideal_length:
        return 0.0
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:k], start=1)
        if doc_id in relevant
    )
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_length + 1))
    return dcg / ideal_dcg


def evaluate(
    index: BM25, queries: dict[str, str], qrels: dict[str, set[str]]
) -> dict[str, float | int]:
    if not qrels:
        raise ValueError("At least one qrel is required")
    ranked = {
        query_id: [doc_id for doc_id, _ in index.search(queries[query_id], 100)]
        for query_id in qrels
    }
    return {
        "queries": len(qrels),
        "relevant_pairs": sum(len(relevant) for relevant in qrels.values()),
        "recall@10": sum(
            recall_at_k(ranked[query_id], relevant, 10)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
        "recall@100": sum(
            recall_at_k(ranked[query_id], relevant, 100)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
        "mrr@10": sum(
            reciprocal_rank_at_k(ranked[query_id], relevant, 10)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
        "ndcg@10": sum(
            ndcg_at_k(ranked[query_id], relevant, 10)
            for query_id, relevant in qrels.items()
        )
        / len(qrels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "corpus.jsonl")
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "queries.jsonl")
    parser.add_argument(
        "--qrels", type=Path, default=DATA_DIR / "qrels" / "test.tsv"
    )
    args = parser.parse_args()

    documents = read_documents(args.corpus)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    missing_queries = sorted(set(qrels) - set(queries))
    if missing_queries:
        raise ValueError(f"Qrels reference missing queries: {missing_queries[:5]}")
    metrics = evaluate(BM25(documents), queries, qrels)
    print(json.dumps({"documents": len(documents), **metrics}, indent=2))


if __name__ == "__main__":
    main()
