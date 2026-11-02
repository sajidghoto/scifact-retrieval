"""Read-only throughput and tokenizer benchmark for local embedding models."""

import argparse
from collections.abc import Sequence
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time
import sys

import torch
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.db.models import Chunk  # noqa: E402


MODELS = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
}
BATCH_SIZES = (8, 16, 32)
SAMPLE_SIZE = 64


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cached_model_path(model_name: str) -> str | None:
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return None


def _load_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    cached_path = _cached_model_path(model_name)
    model = SentenceTransformer(model_name, device=device)
    load_seconds = time.perf_counter() - started
    return model, load_seconds, cached_path is not None


def _token_lengths(tokenizer, texts: Sequence[str]) -> list[int]:
    encoded = tokenizer(
        list(texts), add_special_tokens=True, truncation=False,
        padding=False, return_attention_mask=False,
    )
    return [len(ids) for ids in encoded["input_ids"]]


def _percentile(values: Sequence[int], percentile: float) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[
        int(percentile) - 1
    ]


def _throughput(model, texts: Sequence[str], batch_size: int) -> dict[str, float]:
    warmup = list(texts[: min(batch_size, len(texts))])
    model.encode(
        warmup,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    started = time.perf_counter()
    model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    seconds = time.perf_counter() - started
    return {
        "seconds": seconds,
        "seconds_per_chunk": seconds / len(texts),
        "chunks_per_second": len(texts) / seconds,
    }


def _get_texts() -> list[str]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            return list(
                session.scalars(select(Chunk.text).order_by(Chunk.id))
            )
    finally:
        engine.dispose()


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=tuple(MODELS))
    args = parser.parse_args(argv)
    model_names = args.model or list(MODELS)
    texts = _get_texts()
    if len(texts) != 6274:
        raise RuntimeError(f"Expected 6274 chunks, found {len(texts)}")
    device = _device()
    print(json.dumps({
        "device": device,
        "torch_version": torch.__version__,
        "cpu_cores": multiprocessing.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "configured_batch_size": os.getenv("EMBEDDING_BATCH_SIZE", "32"),
    }))

    first_model = model_names[0]
    first_model_obj, _, _ = _load_model(first_model, device)
    lengths = _token_lengths(first_model_obj.tokenizer, texts)
    max_length = first_model_obj.max_seq_length
    print(json.dumps({
        "tokenizer_model": first_model,
        "model_max_sequence_length": max_length,
        "token_min": min(lengths),
        "token_median": statistics.median(lengths),
        "token_p90": _percentile(lengths, 90),
        "token_p99": _percentile(lengths, 99),
        "token_max": max(lengths),
        "over_model_limit": sum(length > max_length for length in lengths),
    }))
    del first_model_obj

    for model_name in model_names:
        model, load_seconds, cached = _load_model(model_name, device)
        print(json.dumps({
            "model": model_name,
            "dimension": MODELS[model_name],
            "weights_cached_before_load": cached,
            "load_seconds": load_seconds,
        }))
        sample = texts[:SAMPLE_SIZE]
        for batch_size in BATCH_SIZES:
            result = _throughput(model, sample, batch_size)
            print(json.dumps({
                "model": model_name,
                "batch_size": batch_size,
                "sample_chunks": SAMPLE_SIZE,
                **result,
            }))
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
