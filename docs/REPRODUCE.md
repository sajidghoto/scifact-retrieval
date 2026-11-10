# Reproduce the SciFact experiments

This guide is the end-to-end Windows PowerShell path from a fresh clone through retrieval and the full Qwen answer evaluation. Run commands from the repository root unless stated otherwise. The raw SciFact files, embeddings, frozen hybrid configuration, train candidate cache, and retrieval result files are included in the repository. The Qwen generation outputs are produced separately in Colab and are not included.

## 1. Clone and set up Python

Install Git, Python, and Docker Desktop first. Start Docker Desktop, then:

```powershell
git clone https://github.com/sajidghoto/scifact-retrieval.git
Set-Location .\scifact-retrieval
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. `.env.example` contains local database defaults; keep `.env` private and set a different database password if needed.

## 2. Create and populate the database

```powershell
docker compose up -d db
docker compose exec db psql -U scifact -d scifact -c "CREATE EXTENSION IF NOT EXISTS vector;"
alembic -c backend\alembic.ini upgrade head
python .\scripts\ingest\ingest_scifact.py
```

Ingestion loads the bundled SciFact corpus, queries, relevance judgments (qrels), and 256-token chunks with 64-token overlap. Check the imported data:

```powershell
python .\scripts\verify\check_db.py
```

## 3. Import the included embeddings

The repository includes embeddings generated with `BAAI/bge-base-en-v1.5` (768 dimensions), so Colab embedding generation is not needed to reproduce the reported retrieval run. Validate and then import them:

```powershell
python .\scripts\embed\import_embeddings.py --embeddings .\embeddings.npy --ids .\ids.npy --metadata .\embeddings_meta.json --dry-run
python .\scripts\embed\import_embeddings.py --embeddings .\embeddings.npy --ids .\ids.npy --metadata .\embeddings_meta.json
```

To independently check a sample against the local embedding model, run:

```powershell
python .\scripts\embed\parity_check.py
```

Optional: regenerate embeddings by exporting chunks with `python .\scripts\embed\export_chunks.py`, following [notebooks/colab_embed.ipynb](../notebooks/colab_embed.ipynb), and importing the downloaded arrays using the same dry-run/import commands with their filenames. This requires a Colab GPU and downloads the embedding model.

## 4. Reproduce retrieval and statistical results

The experiment tunes fusion on the 809 train queries and evaluates the selected configuration once on the separate 300-query test split. The committed train candidate cache lets you rerun tuning without recomputing train retrieval:

```powershell
python .\scripts\eval\tune_hybrid.py --config .\config\hybrid_reproduced.json
```

The selected method, alpha, and depth should match [`config/hybrid_final.json`](../config/hybrid_final.json). The command writes the tuning tables under `results\` and the separate `hybrid_reproduced.json`, leaving the committed frozen configuration unchanged.

Run each test evaluation (vector modes load the sentence-transformer query model):

```powershell
python .\scripts\eval\run_eval.py --retriever bm25
python .\scripts\eval\run_eval.py --retriever vector
python .\scripts\eval\run_eval.py --retriever vector --no-query-prefix
python .\scripts\eval\run_eval.py --retriever hybrid --config .\config\hybrid_reproduced.json
python .\scripts\eval\bootstrap.py
```

Reports and per-query rankings are written under `results\`. Bootstrap uses 10,000 paired query resamples with seed `20261103`. Do not tune against the test qrels.

To independently regenerate the train candidate cache rather than using the committed one, choose a new output path and then tune from it:

```powershell
python .\scripts\eval\cache_train_candidates.py --cache .\results\cache\train_candidates_rebuilt.json
python .\scripts\eval\tune_hybrid.py --cache .\results\cache\train_candidates_rebuilt.json --config .\config\hybrid_reproduced.json
```

This step embeds train queries and searches the database; it may take a while. It does not use test qrels for tuning.

## 5. Generate Qwen answer prompts

After the test BM25 and hybrid commands have produced their per-query rankings, export the 600 prompts (300 test queries x 2 retrievers):

```powershell
python .\scripts\generation\export_prompts.py
```

Upload both `data\exports\generation_prompts.jsonl` and `data\exports\generation_prompts.meta.json` to [notebooks/colab_generate.ipynb](../notebooks/colab_generate.ipynb). In Colab, select a T4 GPU, run the notebook cells in order, and enter `Qwen/Qwen3-4B` when prompted. The notebook verifies prompt checksums and output keys, saves resumable segments, and downloads `generations.jsonl` and `generations_meta.json`.

Copy both downloaded files into `data\exports\` in the cloned repository. Keep the metadata file: it records the exact Hugging Face model revision and runtime details used.

## 6. Validate, import, and evaluate Qwen outputs

```powershell
python .\scripts\generation\import_generations.py --dry-run
python .\scripts\generation\import_generations.py
```

Copy `model_id` and `revision_hash` from `data\exports\generations_meta.json` into the model argument:

```powershell
python .\scripts\eval\run_generation_eval.py --cache-only --run test --model "Qwen/Qwen3-4B@<revision_hash>"
```

`--cache-only` makes no API calls. The command writes `results\generation_eval.json` and `results\generation_eval.md`; inspect those reports for the answer quality metrics. The cache and Colab output files are local artifacts, not tracked experiment inputs.

## What the experiment measures

- **BM25:** lexical retrieval baseline.
- **Vector:** exact cosine retrieval using BGE chunk embeddings and maximum chunk similarity; the default query includes the documented retrieval prefix. The no-prefix run isolates that choice.
- **Hybrid:** min-max normalized BM25 and vector scores combined with train-selected weighted fusion (alpha 0.7, depth 50).
- **Generation:** Qwen answers from each retriever’s top-five documents, evaluated with deterministic validity, grounding, citation, and abstention metrics. It is not a human evaluation.

See [EVALUATION.md](EVALUATION.md) for dataset counts, metric definitions, results, and limitations; [DECISIONS.md](DECISIONS.md) for design choices.

## Reproducibility notes

- `backend/requirements.txt` uses unpinned dependencies. For closer environment matching, record Python, package, PostgreSQL, Colab runtime, and GPU versions.
- The Qwen notebook records the resolved model revision and library versions in `generations_meta.json`. Model/runtime changes can affect generated text even when the prompts are unchanged.
- `.env`, downloaded Colab outputs, prompt exports, and generation caches should remain local and should not be committed.
- Run the automated tests with `python -m pytest`.
