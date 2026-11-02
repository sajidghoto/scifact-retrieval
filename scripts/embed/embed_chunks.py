"""Resumably embed chunks with a local SentenceTransformers model."""

import argparse
from collections.abc import Callable, Sequence
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Protocol

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.db.models import Chunk  # noqa: E402
from scripts.embed.config import EmbeddingConfig  # noqa: E402


class EmbeddingClient(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...

    def token_count(self, text: str) -> int:
        ...


class AuthenticationError(RuntimeError):
    """Raised when an embedding provider rejects credentials."""


class InvalidEmbeddingError(ValueError):
    """Raised when a provider returns an unusable vector."""


def estimate_tokens(texts: Sequence[str]) -> int:
    """Estimate tokens without loading or calling the embedding model."""
    return sum(len(text.split()) for text in texts)


def process_batches(
    session: Session,
    rows: Sequence[Chunk],
    client: EmbeddingClient,
    config: EmbeddingConfig,
    batch_size: int,
) -> tuple[int, int]:
    total_tokens = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        token_counts = [(row.id, client.token_count(row.text)) for row in batch]
        oversized = [
            (row_id, tokens)
            for row_id, tokens in token_counts
            if tokens > config.token_limit
        ]
        if oversized:
            details = ", ".join(
                f"{row_id} ({tokens})" for row_id, tokens in oversized
            )
            raise ValueError(
                f"Chunks exceed token limit {config.token_limit}: {details}"
            )
        vectors = _retry(
            lambda: client.embed([row.text for row in batch]),
            config.max_retries,
        )
        validated = validate_vectors(vectors, len(batch), config.dimension)
        for row, vector in zip(batch, validated):
            session.execute(
                update(Chunk)
                .where(Chunk.id == row.id, Chunk.embedding.is_(None))
                .values(embedding=vector, embedding_model=config.model)
            )
        session.commit()
        total_tokens += sum(tokens for _, tokens in token_counts)
    return len(rows), total_tokens


class SentenceTransformerClient:
    def __init__(self, model_name: str, token_limit: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "sentence-transformers is required; install backend/requirements.txt"
            ) from error
        self.model = SentenceTransformer(model_name)
        self.tokenizer = self.model.tokenizer
        self.token_limit = token_limit

    def token_count(self, text: str) -> int:
        encoded = self.tokenizer(
            text, add_special_tokens=True, truncation=False
        )
        return len(encoded["input_ids"])

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return self.model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=False
        )


def _is_auth_error(error: Exception) -> bool:
    return isinstance(error, AuthenticationError) or error.__class__.__name__ in {
        "AuthenticationError",
        "InvalidApiKeyError",
    }


def _is_retryable(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    return (
        status_code == 429
        or isinstance(error, (TimeoutError, ConnectionError, OSError))
        or error.__class__.__name__ in {"RateLimitError", "TransientError"}
        or isinstance(status_code, int) and 500 <= status_code < 600
    )


def _retry(
    operation: Callable[[], Sequence[Sequence[float]]],
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> Sequence[Sequence[float]]:
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as error:
            if _is_auth_error(error) or attempt == max_retries:
                raise
            if not _is_retryable(error):
                raise
            delay = min(60.0, 2**attempt) + random.uniform(0, 0.25)
            sleep(delay)
    raise AssertionError("unreachable")


def validate_vectors(
    vectors: Sequence[Sequence[float]], expected_count: int, dimension: int
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise InvalidEmbeddingError(
            f"Expected {expected_count} vectors, received {len(vectors)}"
        )
    validated: list[list[float]] = []
    for index, vector in enumerate(vectors):
        values = [float(value) for value in vector]
        if len(values) != dimension:
            raise InvalidEmbeddingError(
                f"Vector {index} has dimension {len(values)}, expected {dimension}"
            )
        if not all(math.isfinite(value) for value in values):
            raise InvalidEmbeddingError(f"Vector {index} contains non-finite values")
        if float(np.linalg.norm(values)) == 0:
            raise InvalidEmbeddingError(f"Vector {index} has zero norm")
        validated.append(values)
    return validated


def main(
    client: EmbeddingClient | None = None,
    argv: Sequence[str] | None = None,
) -> int:
    load_dotenv(ROOT_DIR / ".env")
    config = EmbeddingConfig.from_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args(argv)
    batch_size = args.batch_size or config.batch_size
    if batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--batch-size and --limit must be greater than zero")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")

    engine = create_engine(database_url)
    started = time.monotonic()
    total_tokens = 0
    total = 0
    try:
        with Session(engine) as session:
            pending_query = select(Chunk).where(
                Chunk.embedding.is_(None)
            ).order_by(Chunk.id)
            if args.limit is not None:
                pending_query = pending_query.limit(args.limit)
            rows = list(session.scalars(pending_query))
            if args.dry_run:
                total_tokens = estimate_tokens([row.text for row in rows])
                print(f"Would embed {len(rows)} chunks")
                print(f"Estimated tokens (whitespace): {total_tokens}")
                print("Estimated API cost: $0 (local inference; compute costs excluded)")
                return 0
            client = client or SentenceTransformerClient(
                config.model, config.token_limit
            )
            total, total_tokens = process_batches(
                session, rows, client, config, batch_size
            )
            if total:
                print(f"Embedded {total}/{len(rows)} chunks")
    finally:
        engine.dispose()
    elapsed = time.monotonic() - started
    print(f"Total tokens: {total_tokens}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print("Estimated API cost: $0 (local inference; compute costs excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
