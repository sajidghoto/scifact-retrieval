import logging

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError
from types import SimpleNamespace

from app import main


def test_cached_bm25_matches_uncached_results(monkeypatch):
    documents = [
        SimpleNamespace(doc_id="doc-a", title="Alpha", text="cell biology"),
        SimpleNamespace(doc_id="doc-b", title="Beta", text="cell chemistry"),
        SimpleNamespace(doc_id="doc-c", title="Gamma", text="unrelated"),
    ]

    class FakeSession:
        def scalars(self, _statement):
            return documents

    uncached = main.BM25(
        [
            main.DocumentRecord(document.doc_id, document.title, document.text)
            for document in documents
        ]
    )
    monkeypatch.setattr(main, "_bm25_retriever", None)
    cached = main._get_bm25_retriever(FakeSession())

    assert cached.search("cell biology", 3) == uncached.search("cell biology", 3)
    assert main._get_bm25_retriever(FakeSession()) is cached


def test_vector_model_failure_is_logged_and_specific(monkeypatch, caplog):
    class FailingEmbedder:
        def __init__(self, _model):
            raise OSError("paging file too small")

    monkeypatch.setattr(main, "_vector_embedder", None)
    monkeypatch.setattr(
        "scripts.eval.run_eval.SentenceTransformerQueryEmbedder",
        FailingEmbedder,
    )
    monkeypatch.setattr(
        "scripts.embed.config.EmbeddingConfig.from_environment",
        classmethod(lambda cls: type("Config", (), {"model": "test-model"})()),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as raised:
            main.search_documents(
                session=object(),
                query="test query",
                limit=1,
                mode="vector",
            )

    assert raised.value.status_code == 503
    assert "paging file too small" in raised.value.detail
    assert "Vector model or dependency failure" in caplog.text


def test_vector_database_failure_is_logged_and_specific(monkeypatch, caplog):
    class Embedder:
        def __init__(self, _model):
            pass

        def embed(self, _text):
            return [1.0]

    class FailingSession:
        def scalar(self, _statement):
            raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(main, "_vector_embedder", Embedder("test-model"))
    monkeypatch.setattr(
        "scripts.embed.config.EmbeddingConfig.from_environment",
        classmethod(lambda cls: type("Config", (), {"model": "test-model"})()),
    )

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as raised:
            main.search_documents(
                session=FailingSession(),
                query="test query",
                limit=1,
                mode="vector",
            )

    assert raised.value.status_code == 503
    assert "database unavailable" in raised.value.detail
    assert "Vector retrieval database failure" in caplog.text
