"""Evaluate BM25 or exact pgvector retrieval on the SciFact test qrels."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.db.models import Document  # noqa: E402
from app.retrieval import DocumentRecord  # noqa: E402
from app.vector_retrieval import VectorRetriever  # noqa: E402
from app.hybrid_retrieval import HybridRetriever  # noqa: E402
from scripts.embed.config import EmbeddingConfig  # noqa: E402
from scripts.eval.baseline import (  # noqa: E402
    read_documents,
    read_queries,
    read_qrels,
)
from scripts.eval.metrics import evaluate_rankings  # noqa: E402
from scripts.eval.retrievers import BM25Retriever  # noqa: E402


class SentenceTransformerQueryEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=False
        )[0]


def percentile(values: list[float], value: float) -> float:
    return float(statistics.quantiles(values, n=100)[value - 1]) if len(values) >= 2 else values[0]


def evaluate_bm25(args, queries, qrels):
    documents = read_documents(args.corpus)
    retriever = BM25Retriever(documents)
    timings = []
    rankings = {}
    per_query = []
    for query_id in qrels:
        started = time.perf_counter()
        results = retriever.search(queries[query_id], 100)
        elapsed = (time.perf_counter() - started) * 1000
        timings.append(elapsed)
        ranked = [doc_id for doc_id, _ in results]
        rankings[query_id] = ranked
        per_query.append({
            "query_id": query_id,
            "top10_doc_ids": ranked[:10],
            "hit_at_10": bool(set(ranked[:10]) & qrels[query_id]),
            "latency_ms": {"total": elapsed},
        })
    return rankings, per_query, timings, {}


def evaluate_vector(args, queries, qrels):
    load_dotenv(ROOT_DIR / ".env")
    config = EmbeddingConfig.from_environment()
    database_url = __import__("os").getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    engine = create_engine(database_url)
    embedder = SentenceTransformerQueryEmbedder(config.model)
    rankings = {}
    per_query = []
    embed_timings = []
    db_timings = []
    total_timings = []
    fetch_counts = []
    under_100 = 0
    try:
        with Session(engine) as session:
            retriever = VectorRetriever(
                session,
                embedder,
                use_query_prefix=not args.no_query_prefix,
            )
            for query_id in qrels:
                total_started = time.perf_counter()
                embed_started = time.perf_counter()
                text = queries[query_id]
                vector = embedder.embed(
                    f"{'Represent this sentence for searching relevant passages: ' if not args.no_query_prefix else ''}{text}"
                )
                embed_ms = (time.perf_counter() - embed_started) * 1000
                db_started = time.perf_counter()
                results = retriever.search_vector(vector, 100)
                db_ms = (time.perf_counter() - db_started) * 1000
                total_ms = (time.perf_counter() - total_started) * 1000
                ranked = [doc_id for doc_id, _ in results]
                rankings[query_id] = ranked
                embed_timings.append(embed_ms)
                db_timings.append(db_ms)
                total_timings.append(total_ms)
                fetch_counts.append(retriever.last_fetch_count)
                under_100 += retriever.last_distinct_documents < 100
                per_query.append({
                    "query_id": query_id,
                    "top10_doc_ids": ranked[:10],
                    "hit_at_10": bool(set(ranked[:10]) & qrels[query_id]),
                    "latency_ms": {"embedding": embed_ms, "db_search": db_ms, "total": total_ms},
                })
    finally:
        engine.dispose()
    return rankings, per_query, total_timings, {
        "embedding_p50_ms": percentile(embed_timings, 50),
        "embedding_p95_ms": percentile(embed_timings, 95),
        "db_search_p50_ms": percentile(db_timings, 50),
        "db_search_p95_ms": percentile(db_timings, 95),
        "fetch_chunks": {"min": min(fetch_counts), "max": max(fetch_counts), "mean": statistics.mean(fetch_counts)},
        "queries_under_100_distinct_documents": under_100,
    }


def evaluate_hybrid(args, queries, qrels):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    documents = read_documents(args.corpus)
    bm25 = BM25Retriever(documents)
    load_dotenv(ROOT_DIR / ".env")
    database_url = __import__("os").getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    engine = create_engine(database_url)
    embedder = SentenceTransformerQueryEmbedder(EmbeddingConfig.from_environment().model)
    rankings = {}
    per_query = []
    timings = []
    try:
        with Session(engine) as session:
            vector = VectorRetriever(session, embedder, use_query_prefix=True)
            retriever = HybridRetriever(
                bm25,
                vector,
                method=config["method"],
                candidate_depth=config["depth"],
                **config["parameters"],
            )
            for query_id in qrels:
                started = time.perf_counter()
                results = retriever.search(queries[query_id], 100)
                elapsed = (time.perf_counter() - started) * 1000
                timings.append(elapsed)
                ranked = [doc_id for doc_id, _ in results]
                rankings[query_id] = ranked
                per_query.append({
                    "query_id": query_id,
                    "top10_doc_ids": ranked[:10],
                    "hit_at_10": bool(set(ranked[:10]) & qrels[query_id]),
                    "latency_ms": {"total": elapsed},
                })
    finally:
        engine.dispose()
    return rankings, per_query, timings, {"frozen_config": config}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["bm25", "vector", "hybrid"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT_DIR / "config" / "hybrid_final.json")
    parser.add_argument("--no-query-prefix", action="store_true")
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "corpus.jsonl")
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=DATA_DIR / "qrels" / "test.tsv")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "results")
    args = parser.parse_args(argv)
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    queries = {query_id: queries[query_id] for query_id in qrels}
    rankings, per_query, timings, extra = (
        evaluate_bm25(args, queries, qrels)
        if args.retriever == "bm25"
        else evaluate_vector(args, queries, qrels)
        if args.retriever == "vector"
        else evaluate_hybrid(args, queries, qrels)
    )
    if qrels and not per_query:
        raise RuntimeError(
            f"{args.retriever} evaluation produced zero per-query rows for "
            f"{len(qrels)} queries"
        )
    metrics = evaluate_rankings(rankings, qrels)
    variant = "no_prefix" if args.no_query_prefix else "prefix"
    if args.retriever == "bm25":
        variant = "default"
    if args.retriever == "hybrid":
        variant = "frozen"
    config = EmbeddingConfig.from_environment()
    result = {
        "config": {
            "retriever": args.retriever,
            "model": config.model if args.retriever in {"vector", "hybrid"} else None,
            "dimension": config.dimension if args.retriever in {"vector", "hybrid"} else None,
            "chunk_size": 256,
            "chunk_overlap": 64,
            "query_prefix": None if args.retriever == "bm25" else not args.no_query_prefix,
            "aggregation": "max chunk cosine similarity" if args.retriever in {"vector", "hybrid"} else None,
            "date": datetime.now(timezone.utc).isoformat(),
        },
        "metrics": metrics,
        "latency_ms": {
            "p50": percentile(timings, 50),
            "p95": percentile(timings, 95),
            **extra,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"eval_{args.retriever}_{variant}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / f"per_query_{args.retriever}_{variant}.json").write_text(
        json.dumps(per_query, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
