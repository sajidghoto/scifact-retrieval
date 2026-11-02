# SciFact Retrieval System

A reproducible document-retrieval research project using the BEIR SciFact benchmark.

## Current milestone

- [x] Initial monorepo layout
- [x] Local PostgreSQL + pgvector configuration
- [x] FastAPI health endpoint
- [x] Minimal Next.js page
- [ ] Deploy the application and managed database
- [x] Add database migrations and schema
- [x] Ingest SciFact documents, queries, relevance judgments, and chunks
- [x] Implement and test baseline chunking
- [x] Build and evaluate lexical retrieval baseline
- [x] Expose retrieval and chunk statistics through the API
- [x] Connect the frontend search interface

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

## Ingest SciFact

Run these commands from the repository root. Make sure the root `.env` has a
`DATABASE_URL` and `POSTGRES_PASSWORD` configured for the local database (see
`.env.example`), then start PostgreSQL and enable pgvector:

```powershell
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic -c backend\alembic.ini upgrade head
python .\scripts\ingest\ingest_scifact.py
python .\scripts\verify\check_db.py
```

The ingestion script reads the local SciFact corpus and queries, plus the test
qrels, and upserts them and their chunks in a single transaction. It can be
rerun safely. The verification script performs read-only integrity checks and
exits non-zero if any check fails.

Embed pending chunks locally with `BAAI/bge-base-en-v1.5` (768 dimensions).
The command is resumable and commits each batch independently:

```powershell
python .\scripts\embed\embed_chunks.py --dry-run
python .\scripts\embed\embed_chunks.py --limit 20
```

Use `--batch-size N` to override the configured batch size. The local model
has no API cost; the estimate excludes local compute costs.

## Colab embedding workflow

1. Export every stored chunk without changing its text:
   `python .\scripts\embed\export_chunks.py`.
2. Upload the generated `chunks.parquet` (or `chunks.jsonl` when pyarrow is
   unavailable) and `chunks.meta.json` to
   [notebooks/colab_embed.ipynb](notebooks/colab_embed.ipynb), select a Colab
   T4 runtime, and run every cell. The notebook verifies both checksums,
   preserves text exactly, embeds with `BAAI/bge-base-en-v1.5`, and downloads
   `embeddings.npy`, `ids.npy`, and `embeddings_meta.json`.
3. Copy those three files to a local directory and validate/import them:
   `python .\scripts\embed\import_embeddings.py --embeddings .\embeddings.npy --ids .\ids.npy --metadata .\embeddings_meta.json --dry-run`
   then omit `--dry-run` to write only previously empty rows.
4. Run the read-only parity check:
   `python .\scripts\embed\parity_check.py`.

Start the frontend in a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend calls `http://localhost:8000` by default. Set
`NEXT_PUBLIC_API_URL` before starting Next.js when the API runs elsewhere.
The API allows requests from `http://localhost:3000` by default; set
`FRONTEND_ORIGIN` for another local frontend origin.

The local API provides:

- `GET /health` — process health check
- `GET /stats` — ingested document and chunk counts
- `GET /search?query=<text>&limit=<1-50>` — BM25-ranked document results

The `/stats` and `/search` endpoints require a configured `DATABASE_URL` and
an ingested local database. Deployment remains deferred until this workflow is
reproducible locally.

## Chunking configuration

The baseline uses whitespace-delimited tokens and fixed-size windows:

- Chunk size: **256 tokens**
- Overlap: **64 tokens**
- Step: 192 tokens

Measured on the local 5,183-document corpus, token lengths have a median of
192, a 95th percentile of 341, and a maximum of 1,524. This configuration
produces 6,274 chunks, with 1,005 documents requiring more than one chunk and
at most 8 chunks per document. The chunker emits a final shorter window and
returns no chunks for empty text.

## Reproducibility

Dataset versions, ingestion parameters, schema migrations, and evaluation
settings are recorded as the project develops. These results are for the
checked-in local SciFact files only.

## Retrieval baseline and evaluation

Run the dependency-light BM25 baseline from the repository root:

```powershell
python .\scripts\eval\baseline.py
```

The baseline indexes each document's title followed by its abstract text using
lowercase alphanumeric tokenization. It uses BM25 with `k1=1.2` and `b=0.75`,
returns at most 100 documents, and breaks equal scores by document ID.
Evaluation uses only `qrels/test.tsv`; score `1` is treated as relevant.
Reported metrics are macro-averaged over qrel queries:

- `Recall@k`: fraction of relevant documents retrieved in the top `k`
- `MRR@10`: reciprocal rank of the first relevant result, capped at 10
- `nDCG@10`: binary-gain normalized discounted cumulative gain

Measured results from the local SciFact corpus:

| Documents | Test queries | Relevant pairs | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 5,183 | 300 | 339 | 0.7843 | 0.8826 | 0.6302 | 0.6622 |

These are macro-averaged over the 300 test queries and must not be interpreted
as results for another dataset version or a chunked/vector retrieval system.