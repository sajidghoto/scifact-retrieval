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


def test_generate_uses_top_five_retrieved_documents(monkeypatch):
    results = [
        main.SearchResult(
            doc_id=f"d{index}",
            title=f"Title {index}",
            text=f"Evidence {index}",
            score=1.0,
        )
        for index in range(5)
    ]

    def retrieve(**kwargs):
        assert kwargs["limit"] == 5
        assert kwargs["mode"] == "hybrid"
        return main.SearchResponse(query="claim", mode="hybrid", results=results)

    monkeypatch.setattr(main, "search_documents", retrieve)
    prompts = []

    class Client:
        def generate(self, prompt):
            prompts.append(prompt)
            return (
                '{"verdict":"supports","cited_doc_ids":["d1"],'
                '"rationale":"Document d1 reports the claimed finding."}'
            )

    monkeypatch.setattr(main, "create_gemini_client", lambda: Client())
    response = main.generate_answer(
        main.GenerateRequest(claim="claim", mode="hybrid"),
        session=object(),
    )

    assert response.verdict == "supports"
    assert response.cited_doc_ids == ["d1"]
    assert len(response.results) == 5
    assert "DOCUMENT ID: d1" in prompts[0]
    assert "DOCUMENT ID: d5" not in prompts[0]


def test_generate_rejects_citations_outside_retrieved_context(monkeypatch):
    monkeypatch.setattr(
        main,
        "search_documents",
        lambda **_kwargs: main.SearchResponse(
            query="claim",
            mode="bm25",
            results=[
                main.SearchResult(
                    doc_id="d1",
                    title="Title",
                    text="Evidence",
                    score=1.0,
                )
            ],
        ),
    )

    class Client:
        def generate(self, _prompt):
            return (
                '{"verdict":"supports","cited_doc_ids":["unknown"],'
                '"rationale":"The evidence supports the claim."}'
            )

    monkeypatch.setattr(main, "create_gemini_client", lambda: Client())
    with pytest.raises(HTTPException) as raised:
        main.generate_answer(
            main.GenerateRequest(claim="claim"),
            session=object(),
        )

    assert raised.value.status_code == 502
    assert "outside the context" in raised.value.detail
