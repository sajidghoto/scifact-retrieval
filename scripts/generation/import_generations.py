"""Validate Colab-generated responses and add them to the deterministic cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

from app.generation import PROMPT_VERSION, InvalidAnswerError, parse_response  # noqa: E402

DEFAULT_PROMPTS = ROOT_DIR / "data" / "exports" / "generation_prompts.jsonl"
DEFAULT_SIDECAR = ROOT_DIR / "data" / "exports" / "generation_prompts.meta.json"
DEFAULT_GENERATIONS = ROOT_DIR / "data" / "exports" / "generations.jsonl"
DEFAULT_METADATA = ROOT_DIR / "data" / "exports" / "generations_meta.json"
DEFAULT_CACHE = ROOT_DIR / "results" / "cache" / "generation_test.jsonl"


def _read_jsonl(path: Path, label: str) -> list[dict]:
    rows = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid {label} JSON on line {line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"Invalid {label} row on line {line_number}")
                rows.append(row)
    except OSError as error:
        raise ValueError(f"Could not read {label} file {path}") from error
    return rows


def _read_json(path: Path, label: str) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} file {path}") from error
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a JSON object")
    return result


def validate_generations(
    *,
    prompts_path: Path,
    sidecar_path: Path,
    generations_path: Path,
    metadata_path: Path,
) -> tuple[list[dict], dict, dict]:
    prompt_bytes = prompts_path.read_bytes()
    sidecar = _read_json(sidecar_path, "prompt sidecar")
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    if sidecar.get("sha256") != prompt_sha256:
        raise ValueError("Prompt file checksum does not match its sidecar")
    if not isinstance(sidecar.get("row_count"), int) or sidecar["row_count"] <= 0:
        raise ValueError("Prompt sidecar has an invalid row_count")
    if sidecar["row_count"] != 600:
        raise ValueError(f"Expected 600 prompts in the sidecar, found {sidecar['row_count']}")

    prompt_rows = _read_jsonl(prompts_path, "prompt")
    if len(prompt_rows) != sidecar["row_count"]:
        raise ValueError("Prompt row count does not match its sidecar")
    expected_keys = set()
    for line_number, row in enumerate(prompt_rows, start=1):
        key = row.get("key")
        if (
            not isinstance(key, dict)
            or not isinstance(key.get("claim_id"), str)
            or not isinstance(key.get("retriever"), str)
        ):
            raise ValueError(f"Invalid prompt key on line {line_number}")
        normalized = (key["claim_id"], key["retriever"])
        if normalized in expected_keys:
            raise ValueError(f"Duplicate prompt key {normalized}")
        if row.get("prompt_version") != sidecar.get("prompt_version"):
            raise ValueError(f"Prompt version mismatch on line {line_number}")
        context_ids = row.get("context_doc_ids")
        if (
            not isinstance(context_ids, list)
            or len(context_ids) != 5
            or not all(isinstance(doc_id, str) for doc_id in context_ids)
        ):
            raise ValueError(f"Invalid prompt context on line {line_number}")
        expected_keys.add(normalized)
    if sidecar.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Prompt sidecar has an unsupported prompt_version")

    metadata = _read_json(metadata_path, "generation metadata")
    if metadata.get("prompt_version") != sidecar["prompt_version"]:
        raise ValueError("Generation metadata prompt_version does not match the sidecar")
    if metadata.get("prompts_sha256") != sidecar["sha256"]:
        raise ValueError("Generation metadata prompts_sha256 does not match the sidecar")
    if metadata.get("output_count") != sidecar["row_count"]:
        raise ValueError("Generation metadata output_count does not match the sidecar")
    model_id = metadata.get("model_id")
    revision = metadata.get("revision_hash")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Generation metadata must include a non-empty model_id")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Generation metadata must include a non-empty revision_hash")

    generation_rows = _read_jsonl(generations_path, "generation")
    observed_keys = set()
    for line_number, row in enumerate(generation_rows, start=1):
        key = row.get("key")
        if (
            not isinstance(key, dict)
            or not isinstance(key.get("claim_id"), str)
            or not isinstance(key.get("retriever"), str)
        ):
            raise ValueError(f"Invalid generation key on line {line_number}")
        normalized = (key["claim_id"], key["retriever"])
        if normalized not in expected_keys:
            raise ValueError(f"Generation key {normalized} is not in the prompt export")
        if normalized in observed_keys:
            raise ValueError(f"Duplicate generation key {normalized}")
        output = row.get("raw_output_text")
        if not isinstance(output, str) or not output.strip():
            raise ValueError(f"Empty generation output on line {line_number}")
        if (
            not isinstance(row.get("new_tokens"), int)
            or row["new_tokens"] < 0
            or not isinstance(row.get("hit_max_new_tokens"), bool)
        ):
            raise ValueError(f"Invalid generation metadata on line {line_number}")
        observed_keys.add(normalized)
    missing_keys = expected_keys - observed_keys
    if missing_keys:
        raise ValueError(f"Missing generated keys: {sorted(missing_keys)[:10]}")
    if len(generation_rows) != sidecar["row_count"]:
        raise ValueError(
            f"Expected {sidecar['row_count']} generated outputs, found {len(generation_rows)}"
        )
    return generation_rows, metadata, sidecar


def import_generations(
    *,
    prompts_path: Path = DEFAULT_PROMPTS,
    sidecar_path: Path = DEFAULT_SIDECAR,
    generations_path: Path = DEFAULT_GENERATIONS,
    metadata_path: Path = DEFAULT_METADATA,
    cache_path: Path = DEFAULT_CACHE,
    dry_run: bool = False,
) -> dict:
    generation_rows, metadata, sidecar = validate_generations(
        prompts_path=prompts_path,
        sidecar_path=sidecar_path,
        generations_path=generations_path,
        metadata_path=metadata_path,
    )
    model = f"{metadata['model_id']}@{metadata['revision_hash']}"
    prompt_rows = _read_jsonl(prompts_path, "prompt")
    contexts = {
        (row["key"]["claim_id"], row["key"]["retriever"]): row["context_doc_ids"]
        for row in prompt_rows
    }
    existing_rows = _read_jsonl(cache_path, "generation cache") if cache_path.exists() else []
    existing_by_key = {}
    for line_number, row in enumerate(existing_rows, start=1):
        fields = ("claim_id", "retriever", "prompt_version", "model")
        if any(not isinstance(row.get(field), str) for field in fields):
            raise ValueError(f"Invalid generation cache key on line {line_number}")
        key = tuple(row[field] for field in fields)
        if key in existing_by_key:
            raise ValueError(f"Duplicate existing generation cache key {key}")
        existing_by_key[key] = row

    new_rows = []
    output_counts = Counter()
    for row in generation_rows:
        claim_id = row["key"]["claim_id"]
        retriever = row["key"]["retriever"]
        context_ids = contexts[(claim_id, retriever)]
        raw_output = row["raw_output_text"]
        cache_row = {
            "claim_id": claim_id,
            "retriever": retriever,
            "prompt_version": sidecar["prompt_version"],
            "model": model,
            "raw_response": raw_output,
            "context_doc_ids": context_ids,
        }
        try:
            parsed = parse_response(raw_output, set(context_ids))
            cache_row["parsed"] = {
                "verdict": parsed.verdict,
                "cited_doc_ids": parsed.cited_doc_ids,
                "rationale": parsed.rationale,
            }
        except InvalidAnswerError as error:
            cache_row["parsed"] = None
            cache_row["failure"] = str(error)
        key = (claim_id, retriever, sidecar["prompt_version"], model)
        existing = existing_by_key.get(key)
        if existing is not None:
            if existing != cache_row:
                raise ValueError(f"Conflicting generation cache row already exists for {key}")
            output_counts["already_present"] += 1
        else:
            new_rows.append(cache_row)
        output_counts["hit_max_new_tokens"] += int(row["hit_max_new_tokens"])
        output_counts["contains_think"] += int("<think>" in raw_output)

    if not dry_run and new_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as cache_file:
            for row in new_rows:
                cache_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "model": model,
        "expected_outputs": sidecar["row_count"],
        "imported": len(new_rows),
        "already_present": output_counts["already_present"],
        "hit_max_new_tokens": output_counts["hit_max_new_tokens"],
        "contains_think": output_counts["contains_think"],
        "dry_run": dry_run,
        "cache_path": str(cache_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = import_generations(
        prompts_path=args.prompts,
        sidecar_path=args.sidecar,
        generations_path=args.generations,
        metadata_path=args.metadata,
        cache_path=args.cache,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
