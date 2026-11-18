# Deployment

No hosted deployment is provided. The system runs locally with FastAPI and a PostgreSQL/pgvector container.

## Measured local requirements

The existing single-worker API measurements used Python `http.client`, refreshed `psutil` RSS, and 30 requests per mode.

| Mode | Startup RSS (MB) | RSS after first vector (MB) | RSS after 20 mixed (MB) | Peak RSS (MB) | BM25 p50/p95 (ms) | Vector p50/p95 (ms) | Hybrid p50/p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lazy | 77.01 | 883.87 | 958.91 | 965.23 | 28.83 / 42.06 | 132.98 / 164.69 | 156.95 / 223.00 |
| Preload | 944.79 | 949.71 | 958.43 | 965.61 | 27.01 / 40.33 | 177.64 / 221.29 | 200.43 / 260.88 |

Measured cold-start values were 50,763.90 ms to preload readiness, 37,024.68 ms for model construction, and 5,142.55 ms for BM25 construction. The container image size and container memory-limit behavior were not tested, so no container rows are reported.

## Hosting implications

The measured peak RSS establishes a local memory floor near 965.61 MB for the tested process behavior, before operating-system and database overhead. Lazy mode defers model cost until the first vector request; preload shifts that cost to startup and exposes readiness only after completion. Engineering options for a hosted system would include a smaller or quantized model, or externally hosted query embedding; none is evaluated here.

## Local run

This runs the local API and web UI on Windows with PowerShell. Install Python, Node.js/npm, and Docker Desktop, and start Docker Desktop first. Run commands from the repository root unless noted otherwise.

Create and activate a Python environment, install the backend dependencies, and create the local environment file:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. Keep `.env` private; if you change the example database password, update it in both `POSTGRES_PASSWORD` and `DATABASE_URL`. The relevant variables are:

- `DATABASE_URL`
- `POSTGRES_PASSWORD`
- `SCIFACT_PRELOAD`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`

Start PostgreSQL, create the vector extension, apply the schema, and ingest the bundled SciFact data:

```powershell
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic -c backend\alembic.ini upgrade head
python .\scripts\ingest\ingest_scifact.py
```

To enable vector and hybrid search, also import the embeddings included in the repository:

```powershell
python .\scripts\embed\import_embeddings.py --embeddings .\embeddings.npy --ids .\ids.npy --metadata .\embeddings_meta.json
```

Start the backend in one terminal:

```powershell
uvicorn app.main:app --app-dir backend --reload
```

In a second terminal, install and start the frontend:

```powershell
Set-Location .\frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to search or generate an answer from the top five retrieved documents. The frontend defaults to the API at `http://localhost:8000`; set `NEXT_PUBLIC_API_URL` before starting Next.js if the API runs elsewhere. Answer generation requires `GEMINI_API_KEY` and `GEMINI_MODEL` in the backend `.env`; credentials are never sent to the frontend. The API exposes `GET /health` for process health, `GET /ready` for preload readiness, and interactive endpoint documentation at [http://localhost:8000/docs](http://localhost:8000/docs). `/ready` returns 503 when preload is disabled, incomplete, or failed. BM25 search works after ingestion; vector and hybrid modes require the embedding import and download the query embedding model on first use.
