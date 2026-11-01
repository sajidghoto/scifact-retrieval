# SciFact Retrieval — Copilot Context

Keep this file as concise, current project memory. At the end of work that materially changes project status or decisions, update only the relevant bullets below. Remove completed/stale items; do not duplicate the README or keep a chronological log.

## Current state

- Research project: reproducible document retrieval over BEIR SciFact.
- Backend: FastAPI health endpoint at `backend/app/main.py`; SQLAlchemy models for documents, chunks, queries, and qrels are in `backend/app/db/models.py`.
- Data: local SciFact corpus, queries, and qrels are under `data/raw/scifact/`.
- Ingestion: `scripts/ingest/ingest_scifact.py` was verified against the local and fresh databases: 5,183 documents, 6,274 chunks, 1,109 queries, and 339 test qrels. Integrity checks and rerun idempotency passed.
- Frontend: Next.js search UI connected to the locally verified `/search` and `/stats` endpoints. Embeddings are not implemented.
- Tests: chunking and BM25 unit tests are in `tests/test_chunking.py` and `tests/test_baseline.py`.
- Migration `0638444e2212` includes the nullable `chunks.embedding` pgvector column defined by the model. The local and fresh databases were upgraded and verified at this revision.
- BM25 baseline measured 5,183 documents, 300 queries, 339 relevant_pairs, recall@10 0.784, recall@100 0.883, mrr@10 0.630, ndcg@10 0.662.

## Decisions

- Use PostgreSQL + pgvector, SQLAlchemy 2, and Alembic.
- Keep ingestion separate from the FastAPI app; launch it from the repository root with `python .\scripts\ingest\ingest_scifact.py`.
- Preserve SciFact IDs as strings; ingestion uses PostgreSQL upserts and is intended to be rerunnable.
- Use whitespace-delimited 256-token windows with 64-token overlap; the measured corpus distribution is documented in `readme.md`.

## Next steps

1. Choose one hosted embedding model and match `chunks.embedding` dimension.
2. Embed chunks idempotently.
3. Implement a vector retriever and retriever-agnostic evaluation.

## Useful commands

Run from the repository root; configure `.env` using `.env.example`.

```powershell
python -m pip install -r backend/requirements.txt
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic -c backend\alembic.ini upgrade head
python .\scripts\ingest\ingest_scifact.py
uvicorn app.main:app --app-dir backend --reload
```

See `readme.md` for the development setup and project overview.
