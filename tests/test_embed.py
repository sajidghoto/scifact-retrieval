import sys
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.embed.config import EmbeddingConfig
from scripts.embed.embed_chunks import (
    AuthenticationError,
    InvalidEmbeddingError,
    _retry,
    estimate_tokens,
    process_batches,
    validate_vectors,
)
from scripts.embed.export_chunks import _rows_checksum
from scripts.embed.import_embeddings import load_and_validate, validate_known_ids


class FakeClient:
    def __init__(self, vectors, token_count=2):
        self.vectors = vectors
        self.calls = []
        self.token_count_value = token_count

    def token_count(self, text):
        return self.token_count_value

    def embed(self, texts):
        self.calls.append(list(texts))
        return self.vectors


class FakeSession:
    def __init__(self):
        self.executions = []
        self.commits = 0

    def execute(self, statement):
        self.executions.append(statement)

    def commit(self):
        self.commits += 1


def config(**overrides):
    values = {
        "model": "test-model",
        "dimension": 2,
        "batch_size": 2,
        "max_retries": 2,
        "token_limit": 10,
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


def rows(count):
    return [
        SimpleNamespace(id=index, text=f"chunk {index}", embedding=None)
        for index in range(count)
    ]


def test_process_batches_commits_each_batch():
    session = FakeSession()
    client = FakeClient([[1, 0], [0, 1]])

    total, tokens = process_batches(
        session, rows(4), client, config(), batch_size=2
    )

    assert total == 4
    assert tokens == 8
    assert len(client.calls) == 2
    assert session.commits == 2


def test_resume_processes_only_remaining_rows():
    session = FakeSession()
    client = FakeClient([[1, 0]])

    total, _ = process_batches(
        session, rows(1), client, config(), batch_size=2
    )

    assert total == 1
    assert len(client.calls) == 1
    assert session.commits == 1


def test_retry_retries_transient_error_without_retrying_authentication():
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return [[1, 0]]

    assert _retry(operation, 2, sleep=lambda _: None) == [[1, 0]]
    assert attempts == 3

    with pytest.raises(AuthenticationError):
        _retry(lambda: (_ for _ in ()).throw(AuthenticationError()), 2)


@pytest.mark.parametrize(
    "vectors",
    [[[1, 0, 2]], [[float("nan"), 0]], [[0, 0]]],
)
def test_rejects_bad_vectors(vectors):
    with pytest.raises(InvalidEmbeddingError):
        validate_vectors(vectors, 1, 2)


def test_limit_is_applied_by_the_pending_query():
    session = FakeSession()
    client = FakeClient([[1, 0]])
    assert process_batches(session, rows(1), client, config(), 1)[0] == 1


def test_dry_run_token_estimate_does_not_call_client():
    assert estimate_tokens(["one two", "three"]) == 3


def test_export_checksum_is_stable_and_id_ordered():
    rows = [(2, "second"), (1, "first")]
    assert _rows_checksum(rows) == _rows_checksum(rows)
    assert _rows_checksum(rows) != _rows_checksum(sorted(rows))


def valid_files(tmp_path, *, dimension=2, ids=None, vectors=None, model="test-model"):
    ids = np.asarray([1, 2] if ids is None else ids, dtype=np.int64)
    vectors = np.asarray(
        [[1, 0], [0, 1]] if vectors is None else vectors, dtype=np.float32
    )
    embeddings_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "ids.npy"
    metadata_path = tmp_path / "embeddings_meta.json"
    np.save(embeddings_path, vectors)
    np.save(ids_path, ids)
    metadata_path.write_text(
        json.dumps({"model": model, "dimension": dimension}), encoding="utf-8"
    )
    return embeddings_path, ids_path, metadata_path


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"vectors": [[1, 0, 0], [0, 1, 0]]}, "shape"),
        ({"vectors": [[float("nan"), 0], [0, 1]]}, "non-finite"),
        ({"ids": [1, 1]}, "Duplicate"),
        ({"model": "other-model"}, "model"),
    ],
)
def test_import_validation_failures(tmp_path, kwargs, message):
    paths = valid_files(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        validation_config = config() if "model" in kwargs else config(model="test-model")
        load_and_validate(*paths, validation_config)


def test_import_validation_rejects_unknown_dimension_metadata(tmp_path):
    paths = valid_files(tmp_path, dimension=3)
    with pytest.raises(ValueError, match="dimension"):
        load_and_validate(*paths, config(model="test-model"))


def test_import_validation_rejects_unknown_id():
    with pytest.raises(ValueError, match="Unknown"):
        validate_known_ids(np.asarray([1, 3]), {1, 2})


class FakeImportSession:
    def __init__(self):
        self.executions = 0
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def scalars(self, statement):
        return iter([FakeChunk(1), FakeChunk(2)])

    def execute(self, statement):
        self.executions += 1
        return SimpleNamespace(rowcount=1 if self.executions == 1 else 0)

    def commit(self):
        self.commits += 1


class FakeEngine:
    def dispose(self):
        pass


class FakeChunk(int):
    embedding = None


def test_import_dry_run_makes_no_changes(tmp_path, monkeypatch):
    from scripts.embed import import_embeddings

    paths = valid_files(tmp_path)
    fake_session = FakeImportSession()
    monkeypatch.setattr(import_embeddings, "create_engine", lambda _: FakeEngine())
    monkeypatch.setattr(import_embeddings, "Session", lambda _: fake_session)
    monkeypatch.setattr(
        import_embeddings.EmbeddingConfig,
        "from_environment",
        classmethod(lambda cls: config(model="test-model")),
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    summary = import_embeddings.import_embeddings(*paths, dry_run=True)
    assert summary == {"input": 2, "updated": 2, "skipped": 0}
    assert fake_session.executions == 0
    assert fake_session.commits == 0


def test_import_rerun_skips_already_written_rows(tmp_path, monkeypatch):
    from scripts.embed import import_embeddings

    paths = valid_files(tmp_path)
    fake_session = FakeImportSession()
    monkeypatch.setattr(import_embeddings, "create_engine", lambda _: FakeEngine())
    monkeypatch.setattr(import_embeddings, "Session", lambda _: fake_session)
    monkeypatch.setattr(
        import_embeddings.EmbeddingConfig,
        "from_environment",
        classmethod(lambda cls: config(model="test-model")),
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    first = import_embeddings.import_embeddings(*paths)
    second = import_embeddings.import_embeddings(*paths)
    assert first["updated"] == 1
    assert second["updated"] == 0
    assert second["skipped"] == 2
    assert fake_session.commits == 2
