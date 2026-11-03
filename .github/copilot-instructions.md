# SciFact Retrieval — Copilot Context

Keep this file as concise, current project memory. At the end of work that materially changes project status or decisions, update only the relevant bullets below. Remove completed/stale items; do not duplicate the README or keep a chronological log.

## Current state

- Research project: reproducible document retrieval over BEIR SciFact.
- Backend: FastAPI health endpoint at `backend/app/main.py`; SQLAlchemy models for documents, chunks, queries, and qrels are in `backend/app/db/models.py`.
- Data: local SciFact corpus, queries, and qrels are under `data/raw/scifact/`.
- Ingestion: `scripts/ingest/ingest_scifact.py` was verified against the local and fresh databases: 5,183 documents, 6,274 chunks, 1,109 queries, and 339 test qrels. Integrity checks and rerun idempotency passed.
- Frontend: Next.js search UI connected to the locally verified `/search` and `/stats` endpoints.
- Embeddings: `scripts/embed/embed_chunks.py` uses local `BAAI/bge-base-en-v1.5` vectors with dimension 768; supports `--limit`, `--dry-run`, and `--batch-size`.
- Colab embedding path exists in `notebooks/colab_embed.ipynb`; full embedding/import remains pending.
- Vector retriever and retriever-agnostic evaluation are implemented and measured: prefix vector recall@10 0.8780, recall@100 0.9683, MRR@10 0.7016, nDCG@10 0.7405; no-prefix recall@10 0.8619, recall@100 0.9733, MRR@10 0.7088, nDCG@10 0.7430.
- Tests: 49 tests pass, including generation parsing/retry validation, API failure handling, cached BM25 parity, chunking, BM25, embedding, metrics, bootstrap, fusion, hybrid composition, and vector retrieval tests.
- API vector mode loads the model lazily once per process, BM25 is indexed lazily once per process, and `SCIFACT_PRELOAD=true` enables startup preload with `/ready` reporting completion; model and database failures are logged with specific 503 details.
- API measurements use Python `http.client` and refreshed `psutil` RSS from the actual worker: lazy RSS 77.01/883.87/958.91 MB (startup/first-vector/20-mixed), peak 965.23 MB; preload RSS 944.79/949.71/958.43 MB, peak 965.61 MB. Lazy BM25/vector/hybrid p50/p95 were 28.83/42.06, 132.98/164.69, and 156.95/223.00 ms; preload values were 27.01/40.33, 177.64/221.29, and 200.43/260.88 ms.
- Evaluation reports contain separate query-embedding, DB-search, and total latency p50/p95 values; the README table reflects those fields.
- Migration `9b2c3d4e5f6a` aligns `chunks.embedding` to pgvector dimension 768 and clears prior embeddings.
- BM25 baseline measured 5,183 documents, 300 queries, 339 relevant_pairs, recall@10 0.784, recall@100 0.883, mrr@10 0.630, ndcg@10 0.662.
- Hybrid tuning uses only 809 train queries and freezes its selected configuration in `config/hybrid_final.json`; the single approved test run is complete.
- Train-only hybrid scores: weighted alpha 0.7 at depth 50 gives recall@10 0.8936, recall@100 0.9615, MRR@10 0.7452, and nDCG@10 0.7757.
- Final hybrid test scores: 300 queries, recall@10 0.8963, recall@100 0.9617, MRR@10 0.7233, nDCG@10 0.7590; latency p50/p95 138.58/219.80 ms.
- Hybrid per-query results were regenerated after a missing output-row bug; quality metrics matched the original run to four decimals.
- Evidence-grounded generation is implemented with the versioned `prompts/answer_v1.txt` prompt, strict Gemini JSON validation, resumable raw-response caching, and deterministic evaluation; the real API smoke test is pending Gemini environment configuration.

## Decisions

- Use PostgreSQL + pgvector, SQLAlchemy 2, and Alembic.
- Keep ingestion separate from the FastAPI app; launch it from the repository root with `python .\scripts\ingest\ingest_scifact.py`.
- Preserve SciFact IDs as strings; ingestion uses PostgreSQL upserts and is intended to be rerunnable.
- Use whitespace-delimited 256-token windows with 64-token overlap; the measured corpus distribution is documented in `readme.md`.

## Next steps

1. Configure `GEMINI_API_KEY` and `GEMINI_MODEL`, run the five-claim generation smoke test, then obtain approval before the full hybrid/BM25 run.

## Useful commands

Run from the repository root; configure `.env` using `.env.example`.

```powershell
python -m pip install -r backend/requirements.txt
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic -c backend\alembic.ini upgrade head
python .\scripts\ingest\ingest_scifact.py
uvicorn app.main:app --app-dir backend --reload
python .\scripts\eval\run_eval.py --retriever bm25
python .\scripts\eval\run_eval.py --retriever vector
python .\scripts\eval\run_eval.py --retriever vector --no-query-prefix
```

See `readme.md` for the development setup and project overview.
