"""Cache BM25 and prefixed-vector candidates for SciFact train queries."""

import json
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw" / "scifact"
DEFAULT_CACHE = ROOT_DIR / "results" / "cache" / "train_candidates.json"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.vector_retrieval import VectorRetriever  # noqa: E402
from scripts.embed.config import EmbeddingConfig  # noqa: E402
from scripts.eval.baseline import read_documents, read_queries, read_qrels  # noqa: E402
from scripts.eval.metrics import evaluate_rankings  # noqa: E402
from scripts.eval.run_eval import SentenceTransformerQueryEmbedder  # noqa: E402
from scripts.eval.retrievers import BM25Retriever  # noqa: E402


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--corpus", type=Path, default=DATA_DIR / "corpus.jsonl")
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "queries.jsonl")
    parser.add_argument("--qrels", type=Path, default=DATA_DIR / "qrels" / "train.tsv")
    args = parser.parse_args()
    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    missing = sorted(set(qrels) - set(queries))
    if missing:
        raise ValueError(f"Qrels reference missing queries: {missing[:5]}")
    query_ids = sorted(qrels)
    print(f"Train queries found: {len(query_ids)}", flush=True)

    cache = {"version": 1, "depth": 100, "queries": {}}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))
    cached = cache.setdefault("queries", {})
    pending = [query_id for query_id in query_ids if query_id not in cached]
    if pending:
        load_dotenv(ROOT_DIR / ".env")
        database_url = __import__("os").getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("DATABASE_URL is not set")
        documents = read_documents(args.corpus)
        bm25 = BM25Retriever(documents)
        config = EmbeddingConfig.from_environment()
        embedder = SentenceTransformerQueryEmbedder(config.model)
        engine = create_engine(database_url)
        started = time.perf_counter()
        try:
            with Session(engine) as session:
                vector = VectorRetriever(session, embedder, use_query_prefix=True)
                for position, query_id in enumerate(pending, start=1):
                    query = queries[query_id]
                    bm25_results = bm25.search(query, 100)
                    vector_values = embedder.embed(
                        f"Represent this sentence for searching relevant passages: {query}"
                    )
                    vector_results = vector.search_vector(vector_values, 100)
                    cached[query_id] = {
                        "bm25": [[doc_id, score] for doc_id, score in bm25_results],
                        "vector": [[doc_id, score] for doc_id, score in vector_results],
                    }
                    args.cache.parent.mkdir(parents=True, exist_ok=True)
                    args.cache.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
                    print(f"Cached {position}/{len(pending)} train queries", flush=True)
        finally:
            engine.dispose()
        print(f"Cache time seconds: {time.perf_counter() - started:.2f}", flush=True)
    else:
        print("Cache is complete; no model load required.", flush=True)

    bm25_rankings = {query_id: [x[0] for x in cached[query_id]["bm25"]] for query_id in query_ids}
    vector_rankings = {query_id: [x[0] for x in cached[query_id]["vector"]] for query_id in query_ids}
    print(json.dumps({
        "train_queries": len(query_ids),
        "bm25": evaluate_rankings(bm25_rankings, qrels),
        "vector_prefix": evaluate_rankings(vector_rankings, qrels),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
