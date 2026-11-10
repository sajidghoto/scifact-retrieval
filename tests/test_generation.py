import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "backend"))

from app.generation import (
    GeminiAnswerClient,
    InvalidAnswerError,
    QuotaExhaustedError,
    build_prompt,
    parse_response,
)


def test_valid_response_is_parsed():
    result = parse_response(
        '{"verdict":"supports","cited_doc_ids":["d1"],"rationale":"The abstract reports the claimed result."}',
        {"d1"},
    )
    assert result.verdict == "supports"
    assert result.cited_doc_ids == ["d1"]


@pytest.mark.parametrize(
    "raw",
    ["not json", '{"verdict":"maybe","cited_doc_ids":[],"rationale":"x"}'],
)
def test_invalid_json_and_verdict_fail(raw):
    with pytest.raises(InvalidAnswerError):
        parse_response(raw, {"d1"})


def test_hallucinated_citation_fails():
    with pytest.raises(InvalidAnswerError, match="outside"):
        parse_response(
            '{"verdict":"supports","cited_doc_ids":["d2"],"rationale":"Evidence."}',
            {"d1"},
        )


def test_empty_context_requires_insufficient_answer():
    prompt = build_prompt("A claim", [])
    assert "No documents" in prompt
    assert parse_response(
        '{"verdict":"insufficient","cited_doc_ids":[],"rationale":"The context is empty."}',
        set(),
    ).verdict == "insufficient"


def test_client_retries_429_with_backoff():
    class Error(Exception):
        status_code = 429

    class Models:
        def __init__(self):
            self.calls = 0

        def generate_content(self, **_kwargs):
            self.calls += 1
            if self.calls < 3:
                raise Error("temporary rate limit")
            return SimpleNamespace(text='{"ok": true}')

    models = Models()
    delays = []
    client = GeminiAnswerClient(
        SimpleNamespace(models=models),
        "gemini-test",
        base_delay=1,
        random_fn=lambda: 0,
        sleep_fn=delays.append,
    )
    assert json.loads(client.generate("prompt")) == {"ok": True}
    assert models.calls == 3
    assert delays == [0.5, 1.0]


def test_client_stops_on_daily_quota():
    class Error(Exception):
        status_code = 429

    class Models:
        def generate_content(self, **_kwargs):
            raise Error("daily quota exceeded")

    client = GeminiAnswerClient(SimpleNamespace(models=Models()), "gemini-test")
    with pytest.raises(QuotaExhaustedError):
        client.generate("prompt")
