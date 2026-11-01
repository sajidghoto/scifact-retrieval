"""Deterministic token-window chunking for SciFact documents."""

from collections.abc import Iterator
import re


CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
_TOKEN_PATTERN = re.compile(r"\S+")


def tokenize(text: str) -> list[str]:
    """Split text into whitespace-delimited tokens."""
    return _TOKEN_PATTERN.findall(text)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> Iterator[str]:
    """Yield fixed-size token windows, including a final shorter window."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and less than chunk_size")

    tokens = tokenize(text)
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        yield " ".join(window)
        if start + chunk_size >= len(tokens):
            break
