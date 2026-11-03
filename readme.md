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
- [x] Add leakage-free hybrid retrieval tuning on the SciFact train split

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

The API lazily loads the embedding model and builds the BM25 index once per
process. Set `SCIFACT_PRELOAD=true` to load both at startup; when enabled,
`GET /ready` reports readiness after preload completes. `/health` remains a
cheap process health check.

Single-worker API measurement using Python `http.client`, `psutil` RSS refreshed
on every sample, and 10 distinct queries:

| Loading | RSS after startup (MB) | RSS after first vector (MB) | RSS after 20 mixed (MB) | Peak RSS (MB) | BM25 p50/p95 (ms) | Vector p50/p95 (ms) | Hybrid p50/p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lazy | 77.01 | 883.87 | 958.91 | 965.23 | 28.83 / 42.06 | 132.98 / 164.69 | 156.95 / 223.00 |
| Preload | 944.79 | 949.71 | 958.43 | 965.61 | 27.01 / 40.33 | 177.64 / 221.29 | 200.43 / 260.88 |

Each mode used 30 requests. Lazy first-vector latency was 44,851.34 ms;
preload readiness took 50,763.90 ms. Model construction measured 37,024.68 ms
and BM25 construction measured 5,142.55 ms.

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

## Retriever evaluation

Run the retriever-agnostic evaluation on the 300 qrel-backed test queries:

```powershell
python .\scripts\eval\run_eval.py --retriever bm25
python .\scripts\eval\run_eval.py --retriever vector
python .\scripts\eval\run_eval.py --retriever vector --no-query-prefix
```

Vector retrieval uses exact pgvector cosine search over chunks, fetches up to
the full embedded corpus until at least 100 distinct documents are available,
and aggregates each document by its maximum chunk similarity. Query text uses
the BGE model-card prefix
[`Represent this sentence for searching relevant passages:`](https://huggingface.co/BAAI/bge-base-en-v1.5#using-sentence-transformers)
only for vector queries; stored chunk text is unchanged. The commands write
summary and per-query JSON files under `results/`.

Hybrid evaluation is prepared with the frozen configuration at
`config/hybrid_final.json`:

```powershell
python .\scripts\eval\run_eval.py --retriever hybrid --config .\config\hybrid_final.json
```

The hybrid protocol caches BM25 and prefixed-vector candidates for the 809 train
queries, tunes only from that cache, and freezes the selected method,
parameters, and candidate depth before the 300-query test evaluation.

Final frozen hybrid test result: recall@10 **0.8963**, recall@100 **0.9617**,
MRR@10 **0.7233**, nDCG@10 **0.7590**, with latency p50/p95 of
**138.58/219.80 ms** over 300 queries. The run used weighted fusion with
alpha 0.7 and candidate depth 50.

The hybrid per-query results were regenerated after fixing a missing output-row
bug; all four quality metrics matched the earlier run to four decimals.

Measured on the local SciFact database with `BAAI/bge-base-en-v1.5`:

| Retriever | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Query embedding p50/p95 (ms) | DB search p50/p95 (ms) | Total p50/p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.7843 | 0.8826 | 0.6302 | 0.6622 | n/a | n/a | 21.83 / 55.75 |
| Vector, query prefix | 0.8780 | 0.9683 | 0.7016 | 0.7405 | 95.52 / 201.48 | 60.57 / 116.44 | 155.82 / 314.51 |
| Vector, no query prefix | 0.8619 | 0.9733 | 0.7088 | 0.7430 | 108.07 / 284.52 | 79.17 / 183.18 | 190.45 / 434.22 |

The vector runs evaluated 300 queries and returned at least 100 distinct
documents for every query. Vector prefix embedding/database/total latency
p50/p95 was 95.52/201.48, 60.57/116.44, and 155.82/314.51 ms. Without the
prefix it was 108.07/284.52, 79.17/183.18, and 190.45/434.22 ms.
BM25 also returned 100 distinct documents for all 300 queries. The earlier
summary table had a typo: its header and rows included only one combined
latency column, omitting the query-embedding column; the generated JSON files
were correct and contained embedding, DB-search, and total latency fields.
The prefix did not improve every metric: no-prefix recall@100, MRR@10, and
nDCG@10 were higher (0.9733, 0.7088, and 0.7430) than the prefix values
(0.9683, 0.7016, and 0.7405).

The exact evaluation commands are:

```powershell
python .\scripts\eval\run_eval.py --retriever bm25
python .\scripts\eval\run_eval.py --retriever vector
python .\scripts\eval\run_eval.py --retriever vector --no-query-prefix
python .\scripts\eval\bootstrap.py
```

The bootstrap outputs are
[`results/bootstrap_test.json`](results/bootstrap_test.json) and
[`results/bootstrap_test.md`](results/bootstrap_test.md). Hybrid failure
analysis is recorded in
[`results/failures_hybrid.md`](results/failures_hybrid.md).

The SQL/Python cosine sanity check maximum score difference was
`1.1920928955078125e-07`, and the ranking order matched. Five fixed-seed
chunk self-queries all returned their own document at rank 1. Failure
analysis is recorded in
[`results/failures_bm25_vs_vector.md`](results/failures_bm25_vs_vector.md).

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
- `GET /ready` — preload readiness; returns 503 when preload is disabled,
  incomplete, or failed
- `GET /stats` — ingested document and chunk counts
- `GET /search?query=<text>&limit=<1-50>` — BM25-ranked document results
- `GET /search?query=<text>&limit=<1-50>&mode=hybrid` — frozen hybrid results

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