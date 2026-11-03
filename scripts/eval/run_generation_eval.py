"""Run deterministic evaluation of cached Gemini evidence-grounded answers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.generation import (  # noqa: E402
    PROMPT_VERSION,
    EvidenceDocument,
    GenerationError,
    QuotaExhaustedError,
    build_prompt,
    create_gemini_client,
    parse_response,
)
from scripts.eval.baseline import read_documents, read_queries, read_qrels  # noqa: E402
from scripts.eval.bootstrap import paired_bootstrap  # noqa: E402


def _load_rankings(path: Path) -> dict[str, list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["query_id"]: row["top10_doc_ids"][:5] for row in rows}


def _load_cache(path: Path) -> dict[tuple[str, str, str, str], dict]:
    if not path.exists():
        return {}
    cache = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                row = json.loads(line)
                key = (
                    row["claim_id"],
                    row["retriever"],
                    row["prompt_version"],
                    row["model"],
                )
            except (KeyError, json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"Invalid generation cache row {line_number}") from error
            cache[key] = row
    return cache


def _append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _answer_metrics(rows: list[dict], qrels: dict[str, set[str]]) -> tuple[dict, dict[str, np.ndarray]]:
    counts = Counter()
    values = Counter()
    per_query = {}
    for row in rows:
        parsed = row.get("parsed")
        context = set(row["context_doc_ids"])
        relevant = qrels[row["claim_id"]]
        has_relevant_context = bool(context & relevant)
        valid = parsed is not None
        grounded = valid and set(parsed["cited_doc_ids"]).issubset(context)
        cited = valid and bool(parsed["cited_doc_ids"])
        cited_set = set(parsed["cited_doc_ids"]) if cited else set()
        if valid:
            counts["valid"] += 1
            values["validity"] += 1
        if grounded:
            counts["grounded"] += 1
            values["grounding"] += 1
        if cited:
            counts["with_citations"] += 1
            precision = len(cited_set & relevant) / len(cited_set)
            recall = len(cited_set & relevant) / len(relevant) if relevant else 0.0
            values["citation_precision"] += precision
            values["citation_recall"] += recall
            values["answered_relevant"] += bool(cited_set & relevant)
        if valid and parsed["verdict"] == "insufficient":
            counts["insufficient"] += 1
        group = "relevant_in_top5" if has_relevant_context else "no_relevant_in_top5"
        per_query[row["claim_id"]] = {
            "validity": float(valid),
            "grounding": float(grounded),
            "citation_precision": float(values["citation_precision"]) if cited else 0.0,
            "citation_recall": float(values["citation_recall"]) if cited else 0.0,
            "answered_relevant": float(bool(cited_set & relevant)) if cited else 0.0,
            "insufficient": float(valid and parsed["verdict"] == "insufficient"),
            "group": group,
        }
    total = len(rows)
    cited_count = counts["with_citations"]
    groups = {}
    for group in ("no_relevant_in_top5", "relevant_in_top5"):
        group_rows = [r for r in rows if per_query[r["claim_id"]]["group"] == group]
        group_insufficient = sum(
            bool(r.get("parsed") and r["parsed"]["verdict"] == "insufficient")
            for r in group_rows
        )
        groups[group] = {
            "queries": len(group_rows),
            "insufficient": group_insufficient,
            "abstention_rate": group_insufficient / len(group_rows) if group_rows else None,
        }
    metrics = {
        "queries": total,
        "validity_rate": {"count": counts["valid"], "denominator": total, "value": counts["valid"] / total},
        "grounding_rate": {"count": counts["grounded"], "denominator": total, "value": counts["grounded"] / total},
        "citation_precision": {"count": cited_count, "denominator": cited_count, "value": values["citation_precision"] / cited_count if cited_count else None},
        "citation_recall": {"count": cited_count, "denominator": cited_count, "value": values["citation_recall"] / cited_count if cited_count else None},
        "answered_with_relevant_citation_rate": {"count": int(values["answered_relevant"]), "denominator": cited_count, "value": values["answered_relevant"] / cited_count if cited_count else None},
        "abstention": groups,
        "failures": total - counts["valid"],
    }
    arrays = {
        "validity_rate": np.asarray([per_query[q]["validity"] for q in per_query]),
        "grounding_rate": np.asarray([per_query[q]["grounding"] for q in per_query]),
        "insufficient_rate": np.asarray([per_query[q]["insufficient"] for q in per_query]),
    }
    return metrics, arrays


def _write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-cap", type=int, default=100)
    parser.add_argument("--daily-limit", type=int, default=1500)
    parser.add_argument("--retrievers", default="hybrid,bm25")
    parser.add_argument("--cache-dir", type=Path, default=ROOT_DIR / "results" / "cache")
    parser.add_argument("--output-json", type=Path, default=ROOT_DIR / "results" / "generation_eval.json")
    parser.add_argument("--output-md", type=Path, default=ROOT_DIR / "results" / "generation_eval.md")
    parser.add_argument("--smoke-output", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.run_cap <= 0 or args.daily_limit <= 0:
        raise ValueError("--run-cap and --daily-limit must be positive")
    load_dotenv(ROOT_DIR / ".env")
    model = os.getenv("GEMINI_MODEL")
    if not model:
        raise RuntimeError("GEMINI_MODEL is not set")
    qrels = read_qrels(DATA_DIR / "qrels" / "test.tsv")
    queries = read_queries(DATA_DIR / "queries.jsonl")
    documents = {doc.doc_id: doc for doc in read_documents(DATA_DIR / "corpus.jsonl")}
    ranking_paths = {
        "hybrid": ROOT_DIR / "results" / "per_query_hybrid_frozen.json",
        "bm25": ROOT_DIR / "results" / "per_query_bm25_default.json",
    }
    selected = list(dict.fromkeys(args.retrievers.split(",")))
    if set(selected) != {"hybrid", "bm25"}:
        raise ValueError("--retrievers must contain hybrid and bm25")
    rankings = {name: _load_rankings(ranking_paths[name]) for name in selected}
    claim_ids = list(qrels)
    if args.limit is not None:
        claim_ids = claim_ids[:args.limit]
    cache_path = args.cache_dir / f"generation_{args.run}.jsonl"
    cache = _load_cache(cache_path)
    keys = [
        (claim_id, retriever, PROMPT_VERSION, model)
        for claim_id in claim_ids
        for retriever in selected
    ]
    missing = [key for key in keys if key not in cache]
    cached_calls = len(keys) - len(missing)
    print(f"Calls made before run: {cached_calls} (cache misses pending: {len(missing)})")
    print(f"Daily calls remaining before run: {args.daily_limit - cached_calls}")
    if args.dry_run:
        print(f"Dry run: API calls needed: {len(missing)}")
        return 0
    if len(missing) > min(args.run_cap, args.daily_limit):
        raise RuntimeError(
            f"Run requires {len(missing)} calls but cap allows {min(args.run_cap, args.daily_limit)}; "
            "increase the cap or resume in another run."
        )
    client = create_gemini_client()
    calls = 0
    for key in missing:
        claim_id, retriever, _, _ = key
        context_ids = rankings[retriever][claim_id][:5]
        context = [
            EvidenceDocument(doc_id, documents[doc_id].title, documents[doc_id].text)
            for doc_id in context_ids
            if doc_id in documents
        ]
        raw_response = ""
        row = {
            "claim_id": claim_id,
            "claim": queries[claim_id],
            "retriever": retriever,
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "date": date.today().isoformat(),
            "context_doc_ids": context_ids,
        }
        try:
            raw_response = client.generate(build_prompt(queries[claim_id], context))
            parsed = parse_response(raw_response, set(context_ids))
            row["parsed"] = {
                "verdict": parsed.verdict,
                "cited_doc_ids": parsed.cited_doc_ids,
                "rationale": parsed.rationale,
            }
        except QuotaExhaustedError:
            raise
        except GenerationError as error:
            row["failure"] = str(error)
            row["parsed"] = None
        row["raw_response"] = raw_response
        _append_cache(cache_path, row)
        cache[key] = row
        calls += 1
    rows_by_retriever = {
        name: [cache[(claim_id, name, PROMPT_VERSION, model)] for claim_id in claim_ids]
        for name in selected
    }
    results = {name: _answer_metrics(rows, qrels) for name, rows in rows_by_retriever.items()}
    examples = []
    for retriever in selected:
        for row in rows_by_retriever[retriever]:
            parsed = row.get("parsed")
            if parsed and set(parsed["cited_doc_ids"]).issubset(set(row["context_doc_ids"])):
                examples.append({
                    "type": "grounded",
                    "retriever": retriever,
                    "claim": row["claim"],
                    "cited_doc_ids": parsed["cited_doc_ids"],
                    "rationale": parsed["rationale"],
                })
            if len(examples) >= 4:
                break
        if len(examples) >= 4:
            break
    failures = []
    for retriever in selected:
        for row in rows_by_retriever[retriever]:
            parsed = row.get("parsed")
            relevant = qrels[row["claim_id"]]
            cited = set(parsed["cited_doc_ids"]) if parsed else set()
            wrong_abstention = bool(
                parsed and parsed["verdict"] == "insufficient"
                and set(row["context_doc_ids"]) & relevant
            )
            if row.get("failure") or wrong_abstention:
                failures.append({
                    "type": "failure",
                    "failure": row.get("failure", "wrong abstention"),
                    "retriever": retriever,
                    "claim": row["claim"],
                    "cited_doc_ids": sorted(cited),
                    "rationale": parsed["rationale"] if parsed else "",
                })
            if len(failures) >= 4:
                break
        if len(failures) >= 4:
            break
    report = {
        "date": date.today().isoformat(),
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "queries": len(claim_ids),
        "api_calls": calls,
        "cache_file": str(cache_path.relative_to(ROOT_DIR)),
        "results": {name: metrics for name, (metrics, _) in results.items()},
        "examples": examples + failures,
        "paired_bootstrap_hybrid_minus_bm25": paired_bootstrap(
            results["hybrid"][1], results["bm25"][1], resamples=10_000
        ),
    }
    _write_report(args.output_json, report)
    args.output_md.write_text(
        "# Generation evaluation\n\n"
        f"- Date: {report['date']}\n- Prompt version: `{PROMPT_VERSION}`\n"
        f"- Model: `{model}`\n- API calls made: {calls}\n\n"
        + "\n".join(
            f"## {name}\n\n```json\n{json.dumps(report['results'][name], indent=2)}\n```\n"
            for name in selected
        )
        + "\n## Examples\n\n"
        + "\n".join(
            f"{index}. **{example['type']} / {example['retriever']}** — "
            f"{example['claim']}  \nCited IDs: `{example['cited_doc_ids']}`  \n"
            f"Rationale: {example['rationale']}"
            for index, example in enumerate(report["examples"], start=1)
        )
        + "\n",
        encoding="utf-8",
    )
    if args.smoke_output:
        smoke = [cache[key] for key in keys[:5]]
        args.smoke_output.write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    print(f"Calls made in run: {calls}; cached calls now available: {len(keys) - len(missing) + calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
