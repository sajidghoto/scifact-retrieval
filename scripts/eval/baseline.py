"""Run the deterministic BM25 retrieval baseline."""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.retrieval import BM25 as _BM25, DocumentRecord  # noqa: E402
from scripts.eval.metrics import (  # noqa: E402
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from scripts.eval.retrievers import BM25Retriever  # noqa: E402

@dataclass(frozen=True)
class Document:
    doc_id: str
    tokens: list[str]


class BM25:
    """Compatibility wrapper for the pre-refactor token-based test API."""

    def __init__(self, documents: list[Document]) -> None:
        self.index = _BM25(
            [
                DocumentRecord(document.doc_id, "", " ".join(document.tokens))
                for document in documents
            ]
        )

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        return self.index.search(query, limit)

K1 = 1.2
B = 0.75


def read_documents(path: Path) -> list[DocumentRecord]:
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
            documents.append(DocumentRecord(doc_id, title, text))
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


def evaluate(
    index: BM25 | BM25Retriever,
    queries: dict[str, str],
    qrels: dict[str, set[str]],
) -> dict[str, float | int]:
    rankings = {
        query_id: [doc_id for doc_id, _ in index.search(queries[query_id], 100)]
        for query_id in qrels
    }
    return evaluate_rankings(rankings, qrels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "corpus.jsonl")
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=DATA_DIR / "qrels" / "test.tsv")
    args = parser.parse_args()
    documents = read_documents(args.corpus)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    missing_queries = sorted(set(qrels) - set(queries))
    if missing_queries:
        raise ValueError(f"Qrels reference missing queries: {missing_queries[:5]}")
    metrics = evaluate(BM25Retriever(documents), queries, qrels)
    print(json.dumps({"documents": len(documents), **metrics}, indent=2))


if __name__ == "__main__":
    main()
