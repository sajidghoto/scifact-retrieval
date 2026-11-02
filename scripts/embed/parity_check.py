"""Compare database embeddings with local CPU model embeddings."""

import argparse
import random
from pathlib import Path
import sys

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "backend"))
sys.path.insert(0, str(ROOT_DIR))

from app.db.models import Chunk  # noqa: E402
from scripts.embed.config import EmbeddingConfig  # noqa: E402
from scripts.embed.embed_chunks import SentenceTransformerClient  # noqa: E402


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
    try:
        with Session(engine) as session:
            rows = list(
                session.scalars(
                    select(Chunk)
                    .where(Chunk.embedding.is_not(None))
                    .order_by(Chunk.id)
                )
            )
            if len(rows) < 20:
                raise ValueError("At least 20 imported embeddings are required")
            sample = random.Random(args.seed).sample(rows, 20)
            client = SentenceTransformerClient(config.model, config.token_limit)
            client.model.to("cpu")
            vectors = np.asarray(client.embed([row.text for row in sample]), dtype=np.float32)
            similarities = []
            for row, vector in zip(sample, vectors):
                stored = np.asarray(row.embedding, dtype=np.float32)
                similarities.append(float(np.dot(vector, stored) / (np.linalg.norm(vector) * np.linalg.norm(stored))))
    finally:
        engine.dispose()
    print(f"Cosine similarity min: {min(similarities):.6f}")
    print(f"Cosine similarity mean: {np.mean(similarities):.6f}")
    print(f"Cosine similarity max: {max(similarities):.6f}")
    below = sum(value < 0.999 for value in similarities)
    print(f"Below 0.999: {below}")
    return 1 if below else 0


if __name__ == "__main__":
    raise SystemExit(main())
