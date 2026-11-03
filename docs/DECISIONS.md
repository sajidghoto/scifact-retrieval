# Decisions

## PostgreSQL and exact pgvector search

**Context:** Retrieval needs durable documents, chunks, qrels, and 768-dimensional vectors.  
**Decision:** Use PostgreSQL with pgvector and exact cosine search.  
**Alternatives considered:** Hosted vector search and approximate indexes.  
**Consequence:** The evaluated system is local and single-node; hosting is not provided.

## BGE-base instead of BGE-large

**Context:** The project needs local query and chunk embeddings.  
**Decision:** Use `BAAI/bge-base-en-v1.5` with 768 dimensions.  
**Alternatives considered:** `BAAI/bge-large-en-v1.5` with 1,024 dimensions.  
**Consequence:** The base model is the checked-in embedding model; no model benchmark result is claimed here.

## Fixed-size chunking

**Context:** SciFact contains short abstracts.  
**Decision:** Use whitespace windows of 256 tokens with 64-token overlap.  
**Alternatives considered:** Sentence or paragraph segmentation.  
**Consequence:** 1,005 of 5,183 documents have at least two chunks, so chunking effects are expected to be limited on this corpus.

## Weighted fusion

**Context:** BM25 and vector retrieval expose complementary rankings.  
**Decision:** Use weighted min-max score fusion with alpha 0.7 and depth 50.  
**Alternatives considered:** RRF and other weighted alpha/depth combinations.  
**Consequence:** On 809 train queries, the selected configuration measured nDCG@10 0.7757; the best listed RRF configuration measured 0.7535.

## Colab GPU embeddings

**Context:** One-off chunk embedding is computationally heavier than local database work.  
**Decision:** Generate embeddings in the supplied Colab T4 notebook and import validated files locally.  
**Alternatives considered:** Recomputing all embeddings on the local CPU.  
**Consequence:** The workflow includes checksum validation and a local CPU parity check.

## Query prefix

**Context:** BGE retrieval supports a search-oriented query prefix.  
**Decision:** Keep the prefix as the default vector and hybrid setting.  
**Alternatives considered:** No prefix.  
**Consequence:** Prefix measured nDCG@10 0.7405 versus 0.7430 without the prefix; it did not improve every metric, but remains the frozen default.

## Per-process BM25 index

**Context:** Rebuilding BM25 on every request would add avoidable work.  
**Decision:** Build BM25 lazily once per process, with optional startup preload.  
**Alternatives considered:** Per-request construction and an external search service.  
**Consequence:** Existing API measurements report BM25 p50/p95 of 28.83/42.06 ms lazy and 27.01/40.33 ms preload.

## No hosted deployment

**Context:** The evidence covers local ingestion, retrieval, and evaluation only.  
**Decision:** Provide local Docker/PostgreSQL and FastAPI instructions, not a hosted deployment.  
**Alternatives considered:** Managed database, hosted API, and production orchestration.  
**Consequence:** Production availability, security, scaling, and operational cost are outside the demonstrated scope.
