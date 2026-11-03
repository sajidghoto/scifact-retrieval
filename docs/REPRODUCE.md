# Reproduce

All commands run from the repository root. Do not commit `.env`, API keys, raw data, generated exports, caches, or embedding arrays.

## Environment and database

Windows PowerShell:

```powershell
Copy-Item .env.example .env
python -m pip install -r backend\requirements.txt
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic -c backend\alembic.ini upgrade head
```

Unix shell:

```bash
cp .env.example .env
python -m pip install -r backend/requirements.txt
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic -c backend/alembic.ini upgrade head
```

The migration command writes schema state to the local database.

## Ingest and integrity

```powershell
python .\scripts\ingest\ingest_scifact.py
python .\scripts\verify\check_db.py
```

```bash
python ./scripts/ingest/ingest_scifact.py
python ./scripts/verify/check_db.py
```

Ingestion writes documents, queries, qrels, and chunks to PostgreSQL. The integrity check is read-only and writes no file.

## Embeddings

Export chunks locally:

```powershell
python .\scripts\embed\export_chunks.py --help
python .\scripts\embed\export_chunks.py
```

```bash
python ./scripts/embed/export_chunks.py --help
python ./scripts/embed/export_chunks.py
```

Upload the generated `chunks.parquet` or `chunks.jsonl` and `chunks.meta.json` to [notebooks/colab_embed.ipynb](../notebooks/colab_embed.ipynb), run it on a Colab T4, and download `embeddings.npy`, `ids.npy`, and `embeddings_meta.json`. This is the only embedding-generation step described here; it is not run by this documentation.

Validate without writing:

```powershell
python .\scripts\embed\import_embeddings.py --embeddings .\embeddings.npy --ids .\ids.npy --metadata .\embeddings_meta.json --dry-run
python .\scripts\embed\parity_check.py --help
```

```bash
python ./scripts/embed/import_embeddings.py --embeddings ./embeddings.npy --ids ./ids.npy --metadata ./embeddings_meta.json --dry-run
python ./scripts/embed/parity_check.py --help
```

The import dry run writes no database changes; omitting `--dry-run` imports previously empty rows. The parity check is read-only and loads the model, so it is intentionally not run during documentation verification.

## Retrieval evaluation

Each command writes a summary and per-query JSON under `results/`. The exact output names are `eval_bm25_default.json` and `per_query_bm25_default.json` for BM25, `eval_vector_prefix.json` and `per_query_vector_prefix.json` for prefixed vector retrieval, `eval_vector_no_prefix.json` and `per_query_vector_no_prefix.json` for no-prefix vector retrieval, and `eval_hybrid_frozen.json` and `per_query_hybrid_frozen.json` for the frozen hybrid run:

```powershell
python .\scripts\eval\run_eval.py --retriever bm25
python .\scripts\eval\run_eval.py --retriever vector
python .\scripts\eval\run_eval.py --retriever vector --no-query-prefix
python .\scripts\eval\run_eval.py --retriever hybrid --config .\config\hybrid_final.json
```

```bash
python ./scripts/eval/run_eval.py --retriever bm25
python ./scripts/eval/run_eval.py --retriever vector
python ./scripts/eval/run_eval.py --retriever vector --no-query-prefix
python ./scripts/eval/run_eval.py --retriever hybrid --config ./config/hybrid_final.json
```

The evaluated protocol is train-only tuning on 809 queries followed by one frozen 300-query test run. These commands load retrieval models where required and were not run for this documentation task.

## Bootstrap and generation

```powershell
python .\scripts\eval\bootstrap.py --help
python .\scripts\eval\bootstrap.py
python .\scripts\eval\run_generation_eval.py --limit 5 --dry-run --run smoke
```

```bash
python ./scripts/eval/bootstrap.py --help
python ./scripts/eval/bootstrap.py
python ./scripts/eval/run_generation_eval.py --limit 5 --dry-run --run smoke
```

Bootstrap writes `results/bootstrap_test.json` and `results/bootstrap_test.md`. Generation writes `results/generation_eval.json` and `results/generation_eval.md`; a non-dry run requires `GEMINI_API_KEY` and `GEMINI_MODEL`. No generation or evaluation was run to produce this documentation.

## Verification performed

The command interfaces were verified by inspecting their argument parsers and using `--help` or a dry-run form where it does not load a model or call Gemini. Long-running ingestion, embedding, retrieval evaluation, bootstrap, and generation jobs were not run.
