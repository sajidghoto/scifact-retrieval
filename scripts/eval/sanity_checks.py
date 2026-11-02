"""Run measured vector-retrieval sanity checks."""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.db.models import Chunk  # noqa: E402
from app.vector_retrieval import VectorRetriever  # noqa: E402
from scripts.embed.config import EmbeddingConfig  # noqa: E402
from scripts.eval.run_eval import SentenceTransformerQueryEmbedder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    load_dotenv(ROOT_DIR / ".env")
    config = EmbeddingConfig.from_environment()
    database_url = __import__("os").getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    engine = create_engine(database_url)
    results = []
    try:
        with Session(engine) as session:
            chunks = list(session.scalars(select(Chunk).where(Chunk.embedding.is_not(None))))
            sample = random.Random(args.seed).sample(chunks, 5)
            embedder = SentenceTransformerQueryEmbedder(config.model)
            retriever = VectorRetriever(session, embedder, use_query_prefix=False)
            for chunk in sample:
                ranked = retriever.search(chunk.text, 1)
                results.append({
                    "chunk_id": chunk.id,
                    "expected_doc_id": chunk.doc_id,
                    "top_doc_id": ranked[0][0] if ranked else None,
                    "rank_1": bool(ranked and ranked[0][0] == chunk.doc_id),
                })
            text = sample[0].text
            vector = embedder.embed(text)
            distance = Chunk.embedding.cosine_distance(vector)
            sql_rows = list(
                session.execute(
                    select(Chunk.id, Chunk.doc_id, distance)
                    .where(Chunk.embedding.is_not(None))
                    .order_by(distance, Chunk.id)
                    .limit(10)
                )
            )
            python_rows = []
            query_norm = np.linalg.norm(vector)
            for row in chunks:
                stored = np.asarray(row.embedding, dtype=np.float32)
                python_rows.append(
                    (row.id, row.doc_id, float(np.dot(vector, stored) / (query_norm * np.linalg.norm(stored))))
                )
            python_rows.sort(key=lambda row: (-row[2], row[0]))
            sql_order = [row[0] for row in sql_rows]
            python_order = [row[0] for row in python_rows[:10]]
            order_matches = sql_order == python_order
    finally:
        engine.dispose()
    print(json.dumps({"own_document_checks": results, "sql_python_order_matches": order_matches}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
