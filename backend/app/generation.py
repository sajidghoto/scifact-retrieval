"""Evidence-grounded answer generation primitives and Gemini client."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

PROMPT_VERSION = "answer_v1"
ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT_DIR / "prompts" / "answer_v1.txt"
VERDICTS = frozenset({"supports", "contradicts", "insufficient"})


@dataclass(frozen=True)
class EvidenceDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ParsedAnswer:
    verdict: str
    cited_doc_ids: list[str]
    rationale: str


class GenerationError(RuntimeError):
    """Base error for a failed generation request or validation."""


class InvalidAnswerError(GenerationError):
    """The model response did not satisfy the required JSON schema."""


class QuotaExhaustedError(GenerationError):
    """The provider reported that the daily quota is exhausted."""


class GeminiResponse(Protocol):
    text: str


class GeminiModels(Protocol):
    def generate_content(self, *, model: str, contents: str, config: Any) -> GeminiResponse:
        ...


class GeminiClient(Protocol):
    models: GeminiModels


def build_prompt(claim: str, documents: list[EvidenceDocument]) -> str:
    """Render the versioned prompt with only the supplied evidence."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    context = "\n\n".join(
        f"DOCUMENT ID: {document.doc_id}\nTITLE: {document.title}\nABSTRACT: {document.text}"
        for document in documents
    )
    return PROMPT_PATH.read_text(encoding="utf-8").format(
        claim=claim,
        context=context or "(No documents were retrieved.)",
    )


def parse_response(raw_response: str, context_doc_ids: set[str]) -> ParsedAnswer:
    """Parse and strictly validate one model response."""
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as error:
        raise InvalidAnswerError("response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise InvalidAnswerError("response JSON must be an object")
    required = {"verdict", "cited_doc_ids", "rationale"}
    if set(payload) != required:
        raise InvalidAnswerError("response must contain exactly the required fields")
    verdict = payload["verdict"]
    cited = payload["cited_doc_ids"]
    rationale = payload["rationale"]
    if verdict not in VERDICTS:
        raise InvalidAnswerError("unknown verdict")
    if not isinstance(cited, list) or not all(isinstance(item, str) for item in cited):
        raise InvalidAnswerError("cited_doc_ids must be a list of strings")
    if len(set(cited)) != len(cited):
        raise InvalidAnswerError("cited_doc_ids must not contain duplicates")
    if not set(cited).issubset(context_doc_ids):
        raise InvalidAnswerError("response cites a document outside the context")
    if not isinstance(rationale, str) or not rationale.strip():
        raise InvalidAnswerError("rationale must be a non-empty string")
    if len(rationale.split()) > 60:
        raise InvalidAnswerError("rationale exceeds 60 words")
    if verdict == "insufficient" and cited:
        raise InvalidAnswerError("insufficient answers must contain no citations")
    return ParsedAnswer(verdict, cited, rationale)


def _status_code(error: Exception) -> int | None:
    for name in ("status_code", "code", "http_status"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    return None


class GeminiAnswerClient:
    """Thin, injectable wrapper around the official Google Gemini SDK."""

    def __init__(
        self,
        client: GeminiClient,
        model: str,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        random_fn=random.random,
        sleep_fn=time.sleep,
    ) -> None:
        if not model.strip():
            raise ValueError("GEMINI_MODEL must be set")
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.random_fn = random_fn
        self.sleep_fn = sleep_fn

    def generate(self, prompt: str) -> str:
        from google.genai import types

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                    ),
                )
                text = getattr(response, "text", None)
                if not isinstance(text, str):
                    raise GenerationError("Gemini response did not contain text")
                return text
            except Exception as error:
                status = _status_code(error)
                message = str(error).lower()
                if status == 429 and any(
                    token in message for token in ("quota", "daily", "resource exhausted")
                ):
                    raise QuotaExhaustedError(
                        "Gemini daily quota is exhausted; stopping generation."
                    ) from error
                retryable = status == 429 or status is not None and status >= 500
                if not retryable or attempt >= self.max_retries:
                    raise GenerationError(
                        f"Gemini request failed after {attempt + 1} attempt(s): {error}"
                    ) from error
                delay = self.base_delay * (2**attempt) * (0.5 + self.random_fn())
                self.sleep_fn(delay)
        raise AssertionError("unreachable")


def create_gemini_client() -> GeminiAnswerClient:
    """Create a configured client without exposing credentials."""
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    if not model:
        raise RuntimeError("GEMINI_MODEL is not set")
    from google import genai

    return GeminiAnswerClient(genai.Client(api_key=api_key), model)
