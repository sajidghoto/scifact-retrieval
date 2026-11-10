import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.generation import PROMPT_VERSION  # noqa: E402
from scripts.eval import run_generation_eval  # noqa: E402
from scripts.generation.export_prompts import export_prompts  # noqa: E402
from scripts.generation.import_generations import import_generations  # noqa: E402


@pytest.fixture
def generation_files(tmp_path):
    prompts_path = tmp_path / "generation_prompts.jsonl"
    sidecar_path = tmp_path / "generation_prompts.meta.json"
    generations_path = tmp_path / "generations.jsonl"
    metadata_path = tmp_path / "generations_meta.json"
    prompts = []
    generations = []
    for query_index in range(300):
        for retriever in ("hybrid", "bm25"):
            key = {"claim_id": f"q{query_index}", "retriever": retriever}
            prompts.append({
                "key": key,
                "prompt": f"Prompt for {key['claim_id']} {retriever}",
                "context_doc_ids": [f"d{index}" for index in range(5)],
                "prompt_version": PROMPT_VERSION,
            })
            generations.append({
                "key": key,
                "raw_output_text": (
                    '{"verdict":"supports","cited_doc_ids":["d0"],'
                    '"rationale":"Evidence is stated in the abstract."}'
                ),
                "new_tokens": 24,
                "hit_max_new_tokens": False,
            })
    prompts_path.write_text(
        "".join(json.dumps(row) + "\n" for row in prompts),
        encoding="utf-8",
    )
    checksum = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    sidecar = {
        "row_count": 600,
        "sha256": checksum,
        "prompt_version": PROMPT_VERSION,
        "export_date": "2026-11-07",
    }
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    generations_path.write_text(
        "".join(json.dumps(row) + "\n" for row in generations),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps({
        "model_id": "Qwen/Qwen3-test",
        "revision_hash": "0123456789abcdef",
        "prompt_version": PROMPT_VERSION,
        "prompts_sha256": checksum,
        "output_count": 600,
    }), encoding="utf-8")
    return prompts_path, sidecar_path, generations_path, metadata_path


def test_export_prompts_is_deterministic_and_checksummed(tmp_path):
    first_path = tmp_path / "first.jsonl"
    first_sidecar = tmp_path / "first.meta.json"
    second_path = tmp_path / "second.jsonl"
    second_sidecar = tmp_path / "second.meta.json"
    first_count, first_checksum = export_prompts(
        output_path=first_path,
        sidecar_path=first_sidecar,
    )
    second_count, second_checksum = export_prompts(
        output_path=second_path,
        sidecar_path=second_sidecar,
    )
    assert first_count == second_count == 600
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_checksum == second_checksum
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first_checksum
    sidecar = json.loads(first_sidecar.read_text(encoding="utf-8"))
    assert sidecar["row_count"] == 600
    assert sidecar["sha256"] == first_checksum
    assert sidecar["prompt_version"] == PROMPT_VERSION


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("missing", "Missing generated keys"),
        ("duplicate", "Duplicate generation key"),
        ("checksum", "prompts_sha256 does not match"),
        ("empty", "Empty generation output"),
    ],
)
def test_import_rejects_invalid_generation_files(
    tmp_path, generation_files, failure, message
):
    prompts, sidecar, generations, metadata = generation_files
    rows = [json.loads(line) for line in generations.read_text(encoding="utf-8").splitlines()]
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1]["key"] = rows[0]["key"]
    elif failure == "checksum":
        meta["prompts_sha256"] = "0" * 64
    elif failure == "empty":
        rows[0]["raw_output_text"] = ""
    generations.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata.write_text(json.dumps(meta), encoding="utf-8")
    cache_path = tmp_path / "cache.jsonl"
    with pytest.raises(ValueError, match=message):
        import_generations(
            prompts_path=prompts,
            sidecar_path=sidecar,
            generations_path=generations,
            metadata_path=metadata,
            cache_path=cache_path,
        )
    assert not cache_path.exists()


def test_import_dry_run_does_not_write_and_import_is_idempotent(
    tmp_path, generation_files
):
    prompts, sidecar, generations, metadata = generation_files
    cache_path = tmp_path / "cache.jsonl"
    report = import_generations(
        prompts_path=prompts,
        sidecar_path=sidecar,
        generations_path=generations,
        metadata_path=metadata,
        cache_path=cache_path,
        dry_run=True,
    )
    assert report["imported"] == 600
    assert not cache_path.exists()

    report = import_generations(
        prompts_path=prompts,
        sidecar_path=sidecar,
        generations_path=generations,
        metadata_path=metadata,
        cache_path=cache_path,
    )
    assert report["imported"] == 600
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 600
    report = import_generations(
        prompts_path=prompts,
        sidecar_path=sidecar,
        generations_path=generations,
        metadata_path=metadata,
        cache_path=cache_path,
    )
    assert report["imported"] == 0
    assert report["already_present"] == 600
    assert len(cache_path.read_text(encoding="utf-8").splitlines()) == 600


def _write_eval_fixture(tmp_path, models):
    (tmp_path / "results").mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    query_ids = ("q1", "q2")
    for retriever in ("hybrid", "bm25"):
        suffix = "hybrid_frozen" if retriever == "hybrid" else "bm25_default"
        ranking_path = tmp_path / "results" / f"per_query_{suffix}.json"
        ranking_path.write_text(
            json.dumps([
                {"query_id": query_id, "top10_doc_ids": ["d1", "d2", "d3", "d4", "d5"]}
                for query_id in query_ids
            ]),
            encoding="utf-8",
        )
    cache_rows = []
    for model in models:
        for query_id in query_ids:
            for retriever in ("hybrid", "bm25"):
                cache_rows.append({
                    "claim_id": query_id,
                    "claim": f"claim {query_id}",
                    "retriever": retriever,
                    "prompt_version": PROMPT_VERSION,
                    "model": model,
                    "context_doc_ids": ["d1", "d2", "d3", "d4", "d5"],
                    "raw_response": (
                        '{"verdict":"supports","cited_doc_ids":["d1"],'
                        '"rationale":"Evidence supports this claim."}'
                    ),
                    "parsed": {
                        "verdict": "supports",
                        "cited_doc_ids": ["d1"],
                        "rationale": "Evidence supports this claim.",
                    },
                })
    (cache_dir / "generation_test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in cache_rows),
        encoding="utf-8",
    )
    return cache_dir


def _configure_eval(monkeypatch, tmp_path):
    monkeypatch.setattr(run_generation_eval, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(run_generation_eval, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(
        run_generation_eval,
        "read_qrels",
        lambda _path: {"q1": {"d1"}, "q2": {"d2"}},
    )
    monkeypatch.setattr(
        run_generation_eval,
        "read_queries",
        lambda _path: {"q1": "claim q1", "q2": "claim q2"},
    )
    monkeypatch.setattr(
        run_generation_eval,
        "read_documents",
        lambda _path: [SimpleNamespace(doc_id=f"d{i}", title="", text="") for i in range(1, 6)],
    )
    monkeypatch.setattr(
        run_generation_eval,
        "create_gemini_client",
        lambda: (_ for _ in ()).throw(AssertionError("API client must not be created")),
    )


def test_cache_only_uses_cache_without_api_calls(tmp_path, monkeypatch):
    _configure_eval(monkeypatch, tmp_path)
    cache_dir = _write_eval_fixture(tmp_path, ["Qwen/Qwen3-test@0123456789abcdef"])
    output_json = tmp_path / "eval.json"
    output_md = tmp_path / "eval.md"
    assert run_generation_eval.main([
        "--cache-only",
        "--cache-dir", str(cache_dir),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
    ]) == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["api_calls"] == 0
    assert report["cache_only"] is True
    assert report["model"] == "Qwen/Qwen3-test@0123456789abcdef"


def test_cache_only_refuses_to_mix_complete_models(tmp_path, monkeypatch):
    _configure_eval(monkeypatch, tmp_path)
    cache_dir = _write_eval_fixture(
        tmp_path,
        ["Qwen/Qwen3-test@revision-a", "Qwen/Qwen3-test@revision-b"],
    )
    with pytest.raises(RuntimeError, match="multiple models"):
        run_generation_eval.main([
            "--cache-only",
            "--cache-dir", str(cache_dir),
            "--output-json", str(tmp_path / "eval.json"),
            "--output-md", str(tmp_path / "eval.md"),
        ])


def test_cache_only_fails_clearly_for_missing_key(tmp_path, monkeypatch):
    _configure_eval(monkeypatch, tmp_path)
    cache_dir = _write_eval_fixture(tmp_path, ["Qwen/Qwen3-test@0123456789abcdef"])
    cache_path = cache_dir / "generation_test.jsonl"
    rows = cache_path.read_text(encoding="utf-8").splitlines()
    cache_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="generation cache key"):
        run_generation_eval.main([
            "--cache-only",
            "--cache-dir", str(cache_dir),
            "--output-json", str(tmp_path / "eval.json"),
            "--output-md", str(tmp_path / "eval.md"),
        ])
