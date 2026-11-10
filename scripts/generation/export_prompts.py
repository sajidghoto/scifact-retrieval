"""Export the frozen test retrieval contexts as generation prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.generation import PROMPT_VERSION, EvidenceDocument, build_prompt  # noqa: E402
from scripts.eval.baseline import read_documents, read_queries, read_qrels  # noqa: E402


def load_rankings(path: Path) -> dict[str, list[str]]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read saved rankings at {path}") from error
    if not isinstance(rows, list):
        raise ValueError(f"Saved rankings at {path} must be a JSON array")
    rankings = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Invalid ranking row in {path}")
        query_id = row.get("query_id")
        doc_ids = row.get("top10_doc_ids")
        if (
            not isinstance(query_id, str)
            or not isinstance(doc_ids, list)
            or not all(isinstance(doc_id, str) for doc_id in doc_ids)
        ):
            raise ValueError(f"Invalid ranking fields in {path}")
        if query_id in rankings:
            raise ValueError(f"Duplicate query {query_id} in saved rankings {path}")
        rankings[query_id] = doc_ids
    return rankings


def export_prompts(
    *,
    output_path: Path,
    sidecar_path: Path,
    queries_path: Path = DATA_DIR / "queries.jsonl",
    corpus_path: Path = DATA_DIR / "corpus.jsonl",
    qrels_path: Path = DATA_DIR / "qrels" / "test.tsv",
    hybrid_rankings_path: Path = ROOT_DIR / "results" / "per_query_hybrid_frozen.json",
    bm25_rankings_path: Path = ROOT_DIR / "results" / "per_query_bm25_default.json",
) -> tuple[int, str]:
    queries = read_queries(queries_path)
    qrels = read_qrels(qrels_path)
    documents = {doc.doc_id: doc for doc in read_documents(corpus_path)}
    ranking_paths = {
        "hybrid": hybrid_rankings_path,
        "bm25": bm25_rankings_path,
    }
    rankings = {name: load_rankings(path) for name, path in ranking_paths.items()}
    expected_claims = set(qrels)
    for retriever, rows in rankings.items():
        if set(rows) != expected_claims:
            raise ValueError(
                f"{retriever} rankings and test qrels do not contain the same query IDs"
            )
    missing_queries = expected_claims - set(queries)
    if missing_queries:
        raise ValueError(f"Test qrels reference missing queries: {sorted(missing_queries)[:5]}")

    serialized_rows = []
    for claim_id in qrels:
        for retriever in ("hybrid", "bm25"):
            context_ids = rankings[retriever][claim_id][:5]
            if len(context_ids) != 5:
                raise ValueError(
                    f"{retriever} ranking for claim {claim_id} has fewer than 5 documents"
                )
            missing_documents = [doc_id for doc_id in context_ids if doc_id not in documents]
            if missing_documents:
                raise ValueError(
                    f"Ranking for claim {claim_id} references missing documents: "
                    f"{missing_documents}"
                )
            context = [
                EvidenceDocument(
                    doc_id=doc_id,
                    title=documents[doc_id].title,
                    text=documents[doc_id].text,
                )
                for doc_id in context_ids
            ]
            row = {
                "key": {"claim_id": claim_id, "retriever": retriever},
                "prompt": build_prompt(queries[claim_id], context),
                "context_doc_ids": context_ids,
                "prompt_version": PROMPT_VERSION,
            }
            serialized_rows.append(json.dumps(row, ensure_ascii=False) + "\n")
    row_count = len(serialized_rows)
    if row_count != 600:
        raise ValueError(f"Expected 600 prompts for the frozen test split, exported {row_count}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(serialized_rows), encoding="utf-8", newline="\n")
    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    sidecar = {
        "row_count": row_count,
        "sha256": checksum,
        "prompt_version": PROMPT_VERSION,
        "export_date": date.today().isoformat(),
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return row_count, checksum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data" / "exports" / "generation_prompts.jsonl",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=ROOT_DIR / "data" / "exports" / "generation_prompts.meta.json",
    )
    args = parser.parse_args(argv)
    row_count, checksum = export_prompts(
        output_path=args.output,
        sidecar_path=args.sidecar,
    )
    print(f"Exported {row_count} prompts to {args.output}")
    print(f"SHA-256: {checksum}")
    print(f"Sidecar: {args.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
