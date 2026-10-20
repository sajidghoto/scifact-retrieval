# SciFact Retrieval System

A reproducible document-retrieval research project using the BEIR SciFact benchmark.

## Current milestone

- [x] Initial monorepo layout
- [x] Local PostgreSQL + pgvector configuration
- [x] FastAPI health endpoint
- [x] Minimal Next.js page
- [ ] Deploy the application and managed database
- [ ] Add database migrations and schema
- [ ] Ingest SciFact documents, queries, and relevance judgments
- [ ] Implement and test baseline chunking
- [ ] Expose chunk statistics through the API

## Architecture

- `backend/`: FastAPI application and database access
- `frontend/`: Next.js interface
- `scripts/ingest/`: reproducible dataset ingestion
- `scripts/eval/`: retrieval evaluation
- `tests/`: automated tests
- `docker-compose.yml`: local PostgreSQL + pgvector

## Development

Requirements: Python, Node.js/npm, and Docker with Compose.

Start the database:

```bash
docker compose up -d db
```

Enable pgvector:

```bash
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Start the API from the repository root:

```bash
python -m pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

Start the frontend in a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

## Chunking configuration

Baseline chunk size and overlap: not yet selected.

The final values will be recorded after token-level measurements on the actual SciFact corpus. The baseline will use fixed-size token windows with overlap, and its chunk-count distribution will be measured before retrieval experiments begin.

## Reproducibility

Dataset versions, ingestion parameters, schema migrations, and evaluation settings will be recorded as the project develops. No retrieval or embedding results are claimed at this stage.