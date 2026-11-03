"""Tune hybrid fusion strictly from a cached SciFact train candidate file."""

import argparse
import json
from datetime import date
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.fusion import reciprocal_rank_fusion, weighted_score_fusion  # noqa: E402
from scripts.eval.baseline import read_queries, read_qrels  # noqa: E402
from scripts.eval.metrics import evaluate_rankings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT_DIR / "results/cache/train_candidates.json")
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=DATA_DIR / "qrels/train.tsv")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "config/hybrid_final.json")
    args = parser.parse_args()
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    query_ids = sorted(qrels)
    if set(query_ids) - set(cache.get("queries", {})):
        raise ValueError("Candidate cache is incomplete for the train qrels")

    rows = []
    for method, parameters in [
        ("rrf", [{"k": k} for k in (10, 30, 60, 100)]),
        ("weighted", [{"alpha": alpha / 10} for alpha in range(1, 10)]),
    ]:
        for depth in (50, 100):
            for parameter in parameters:
                rankings = {}
                for query_id in query_ids:
                    entry = cache["queries"][query_id]
                    bm25 = [tuple(item) for item in entry["bm25"][:depth]]
                    vector = [tuple(item) for item in entry["vector"][:depth]]
                    fused = (
                        reciprocal_rank_fusion(bm25, vector, **parameter)
                        if method == "rrf"
                        else weighted_score_fusion(bm25, vector, **parameter)
                    )
                    rankings[query_id] = [doc_id for doc_id, _ in fused]
                metrics = evaluate_rankings(rankings, qrels)
                rows.append({"method": method, **parameter, "depth": depth, **metrics})

    rows.sort(key=lambda row: (-row["ndcg@10"], row["method"] != "rrf", row["depth"]))
    best = rows[0]
    near = [row for row in rows if best["ndcg@10"] - row["ndcg@10"] <= 0.002]
    simpler = sorted(near, key=lambda row: (row["method"] != "rrf", row["depth"]))[0]
    best = simpler
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "tuning_train.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"Selected: `{best['method']}` with `{best.get('k', best.get('alpha'))}` at depth `{best['depth']}`. "
        "The selected row is within 0.002 nDCG@10 of the best row; the smaller depth was preferred for simplicity.",
        "",
        "| Rank | Method | Parameter | Depth | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows, start=1):
        parameter = row.get("k", row.get("alpha"))
        lines.append(f"| {rank} | {row['method']} | {parameter} | {row['depth']} | {row['ndcg@10']:.4f} | {row['recall@10']:.4f} | {row['recall@100']:.4f} | {row['mrr@10']:.4f} |")
    (args.output_dir / "tuning_train.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.config.parent.mkdir(parents=True, exist_ok=True)
    frozen = {
        "method": best["method"],
        "parameters": {"k": best["k"]} if best["method"] == "rrf" else {"alpha": best["alpha"]},
        "depth": best["depth"],
        "date": date.today().isoformat(),
        "train_query_count": len(query_ids),
    }
    args.config.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train_query_count": len(query_ids), "chosen": best, "frozen_config": frozen, "top5": rows[:5]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
