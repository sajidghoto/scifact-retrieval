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

Create `.env` from `.env.example` and keep credentials there. Relevant variables are:

- `DATABASE_URL`
- `POSTGRES_PASSWORD`
- `SCIFACT_PRELOAD`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`

Start PostgreSQL and the API:

```powershell
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
uvicorn app.main:app --app-dir backend
```

The API exposes `GET /health` for process health and `GET /ready` for preload readiness. `/ready` returns 503 when preload is disabled, incomplete, or failed. Retrieval endpoints are documented in the repository source and require the local database.
