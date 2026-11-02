"""Environment-backed configuration for chunk embedding."""

from dataclasses import dataclass
import os


DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_DIMENSION = 768
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_RETRIES = 3
DEFAULT_TOKEN_LIMIT = 512


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_MODEL
    dimension: int = DEFAULT_DIMENSION
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES
    token_limit: int = DEFAULT_TOKEN_LIMIT

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig":
        return cls(
            model=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
            dimension=_positive_int("EMBEDDING_DIMENSION", DEFAULT_DIMENSION),
            batch_size=_positive_int("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE),
            max_retries=_nonnegative_int(
                "EMBEDDING_MAX_RETRIES", DEFAULT_MAX_RETRIES
            ),
            token_limit=_positive_int("EMBEDDING_TOKEN_LIMIT", DEFAULT_TOKEN_LIMIT),
        )
