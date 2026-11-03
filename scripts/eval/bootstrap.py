"""Paired bootstrap confidence intervals for saved SciFact rankings."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from scripts.eval.baseline import read_qrels
from scripts.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank_at_k

DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
METRICS = ("ndcg@10", "recall@10", "mrr@10")


def per_query_metrics(
    rows: list[dict], qrels: dict[str, set[str]]
) -> tuple[list[str], dict[str, np.ndarray]]:
    by_query = {row["query_id"]: row for row in rows}
    if set(by_query) != set(qrels):
        raise ValueError("Per-query results and qrels do not contain the same query IDs")
    query_ids = list(qrels)
    values = {metric: [] for metric in METRICS}
    for query_id in query_ids:
        ranked = by_query[query_id]["top10_doc_ids"]
        relevant = qrels[query_id]
        values["ndcg@10"].append(ndcg_at_k(ranked, relevant, 10))
        values["recall@10"].append(recall_at_k(ranked, relevant, 10))
        values["mrr@10"].append(reciprocal_rank_at_k(ranked, relevant, 10))
    return query_ids, {metric: np.asarray(scores, dtype=float) for metric, scores in values.items()}


def paired_bootstrap(
    first: dict[str, np.ndarray],
    second: dict[str, np.ndarray],
    *,
    resamples: int = 10_000,
    seed: int = 20261103,
) -> dict[str, dict[str, float | list[float]]]:
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if set(first) != set(second):
        raise ValueError("Systems must provide the same metrics")
    lengths = {len(values) for values in first.values()} | {
        len(values) for values in second.values()
    }
    if len(lengths) != 1 or not lengths or not next(iter(lengths)):
        raise ValueError("Systems must provide equally sized non-empty metric arrays")
    differences = {metric: first[metric] - second[metric] for metric in first}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, next(iter(lengths)), size=(resamples, next(iter(lengths))))
    output = {}
    for metric, values in differences.items():
        sample_means = values[indices].mean(axis=1)
        output[metric] = {
            "mean_difference": float(values.mean()),
            "ci95_percentile": [
                float(np.percentile(sample_means, 2.5)),
                float(np.percentile(sample_means, 97.5)),
            ],
            "share_first_better": float(np.mean(sample_means > 0)),
        }
    return output


def _load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", type=Path, default=DATA_DIR / "qrels" / "test.tsv")
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--output-json", type=Path, default=ROOT_DIR / "results" / "bootstrap_test.json")
    parser.add_argument("--output-md", type=Path, default=ROOT_DIR / "results" / "bootstrap_test.md")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    qrels = read_qrels(args.qrels)
    paths = {
        "bm25": args.results_dir / "per_query_bm25_default.json",
        "vector": args.results_dir / "per_query_vector_prefix.json",
        "hybrid": args.results_dir / "per_query_hybrid_frozen.json",
    }
    metric_data = {}
    for system, path in paths.items():
        query_ids, metric_data[system] = per_query_metrics(_load_rows(path), qrels)
        if len(query_ids) != 300:
            raise ValueError(f"{system} has {len(query_ids)} queries; expected 300")

    summaries = {
        "bm25": json.loads((args.results_dir / "eval_bm25_default.json").read_text(encoding="utf-8")),
        "vector": json.loads((args.results_dir / "eval_vector_prefix.json").read_text(encoding="utf-8")),
        "hybrid": json.loads((args.results_dir / "eval_hybrid_frozen.json").read_text(encoding="utf-8")),
    }
    nDCG_check = {}
    for system in paths:
        mean = float(metric_data[system]["ndcg@10"].mean())
        reported = float(summaries[system]["metrics"]["ndcg@10"])
        nDCG_check[system] = {
            "per_query_mean": mean,
            "reported": reported,
            "matches_to_4_decimals": round(mean, 4) == round(reported, 4),
        }
        if not nDCG_check[system]["matches_to_4_decimals"]:
            raise ValueError(f"{system} nDCG@10 bootstrap check failed")

    comparisons = {}
    for first, second in (("hybrid", "vector"), ("hybrid", "bm25"), ("vector", "bm25")):
        comparisons[f"{first}_vs_{second}"] = paired_bootstrap(
            metric_data[first],
            metric_data[second],
            resamples=args.resamples,
            seed=args.seed,
        )
    output = {
        "method": "recomputed from saved top-10 document IDs using existing metrics",
        "queries": len(qrels),
        "resamples": args.resamples,
        "seed": args.seed,
        "ndcg10_checks": nDCG_check,
        "comparisons": comparisons,
    }
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Paired bootstrap test",
        "",
        "Per-query metrics were recomputed from saved top-10 document IDs using the existing metrics module.",
        f"Paired bootstrap: {args.resamples:,} resamples, seed `{args.seed}`.",
        "",
        "| Comparison | Metric | Mean difference | 95% percentile interval | First better share | Interval excludes zero |",
        "|---|---|---:|---:|---:|---|",
    ]
    for comparison, metrics in comparisons.items():
        for metric in METRICS:
            result = metrics[metric]
            low, high = result["ci95_percentile"]
            lines.append(
                f"| {comparison.replace('_', ' ')} | {metric} | "
                f"{result['mean_difference']:.6f} | [{low:.6f}, {high:.6f}] | "
                f"{result['share_first_better']:.6f} | {'yes' if low > 0 or high < 0 else 'no'} |"
            )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
