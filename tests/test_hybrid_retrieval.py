import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.hybrid_retrieval import HybridRetriever
import scripts.eval.run_eval as run_eval


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        return self.results[:k]


def test_hybrid_composes_retrievers_at_configured_depth():
    bm25 = StubRetriever([("a", 2.0), ("b", 1.0)])
    vector = StubRetriever([("b", 0.9), ("c", 0.8)])
    retriever = HybridRetriever(bm25, vector, method="rrf", candidate_depth=2)
    assert retriever.search("q", 2)
    assert bm25.calls == [("q", 2)]
    assert vector.calls == [("q", 2)]


def test_evaluate_hybrid_returns_per_query_rows_without_model_or_database(monkeypatch, tmp_path):
    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeEngine:
        def dispose(self):
            pass

    class FakeRetriever:
        def search(self, query, k):
            return [("doc-1", 1.0), ("doc-2", 0.5)]

    monkeypatch.setattr(run_eval, "read_documents", lambda path: [])
    monkeypatch.setattr(run_eval, "BM25Retriever", lambda documents: FakeRetriever())
    monkeypatch.setattr(run_eval, "create_engine", lambda url: FakeEngine())
    monkeypatch.setattr(run_eval, "Session", lambda engine: FakeSession())
    monkeypatch.setattr(
        run_eval,
        "SentenceTransformerQueryEmbedder",
        lambda model: object(),
    )
    monkeypatch.setattr(
        run_eval,
        "EmbeddingConfig",
        SimpleNamespace(from_environment=lambda: SimpleNamespace(model="test-model")),
    )
    monkeypatch.setattr(run_eval, "VectorRetriever", lambda *args, **kwargs: FakeRetriever())
    monkeypatch.setattr(run_eval, "HybridRetriever", lambda *args, **kwargs: FakeRetriever())

    config_path = tmp_path / "hybrid.json"
    config_path.write_text(
        '{"method": "weighted", "parameters": {"alpha": 0.7}, "depth": 50}',
        encoding="utf-8",
    )
    args = SimpleNamespace(config=config_path, corpus=tmp_path / "corpus.jsonl")
    queries = {"q1": "first query", "q2": "second query"}
    qrels = {"q1": {"doc-1"}, "q2": {"missing"}}

    rankings, per_query, timings, extra = run_eval.evaluate_hybrid(args, queries, qrels)

    assert set(rankings) == set(queries)
    assert len(per_query) == len(queries)
    assert [row["query_id"] for row in per_query] == ["q1", "q2"]
    assert all(len(row["top10_doc_ids"]) == 2 for row in per_query)
    assert [row["hit_at_10"] for row in per_query] == [True, False]
