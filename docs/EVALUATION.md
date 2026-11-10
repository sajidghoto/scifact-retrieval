# Evaluation

## Question and scope

This report asks whether lexical BM25, exact vector retrieval, and frozen weighted fusion retrieve SciFact evidence under a reproducible local protocol. It also records a deterministic smoke evaluation of answers generated from retrieved top-five contexts. The results do not evaluate production deployment, other datasets, scale, or human judgments.

## Data

| Item | Count |
|---|---:|
| Corpus documents | 5,183 |
| All queries | 1,109 |
| Train queries | 809 |
| Test queries | 300 |
| Train qrels rows | 919 |
| Test qrels rows | 339 |
| Stored chunks | 6,274 |

Chunking uses whitespace-delimited windows of 256 tokens with 64-token overlap. The corpus document token distribution has median 192, p95 341, and maximum 1,524 tokens. There are 1,005 documents with at least two chunks and at most 8 chunks per document. The embedding metadata records 40 chunks over the 512-token model limit. The Colab workflow sets the model maximum sequence length to 512, while the local embedding script rejects over-limit rows instead of silently accepting them.

## Method

BM25 indexes title followed by abstract text using lowercase alphanumeric tokenization, `k1=1.2`, and `b=0.75`. Vector retrieval uses `BAAI/bge-base-en-v1.5` with dimension 768, the query prefix `Represent this sentence for searching relevant passages:`, exact pgvector cosine search, and maximum chunk similarity as the document score. The no-prefix variant removes only that query prefix.

Hybrid retrieval min-max normalizes each candidate list and combines scores as `(1 - alpha) * BM25 + alpha * vector`; absent-list scores are zero. The tuning grid contains weighted alpha values 0.1 through 0.9 at depths 50 and 100, plus RRF values of 10, 30, 60, and 100 at depths 50 and 100. Selection used the best train nDCG@10, with the frozen configuration in `config/hybrid_final.json`: weighted fusion, alpha 0.7, depth 50.

## Protocol and leakage controls

Hybrid parameters were tuned on the train split only (809 queries), then one frozen run was performed on the 300 test queries. Bootstrap intervals are paired over queries, use 10,000 resamples, and use fixed seed 20261103. The hybrid per-query output had a missing-row bug; it was fixed and regenerated, and all four quality metrics reproduced exactly to four decimal places.

Embeddings were computed on a Colab GPU and checked against the local CPU implementation by cosine similarity. The repository records a maximum SQL/Python score difference of 0.0000 to four decimals and matching ranking order; the parity script checks 20 seeded samples and fails when any similarity is below 0.9990.

## Retrieval results

| Retriever | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | Query embedding p50/p95 (ms) | DB search p50/p95 (ms) | Total p50/p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.7843 | 0.8826 | 0.6302 | 0.6622 | — | — | 21.8254 / 55.7514 |
| Vector, prefix | 0.8780 | 0.9683 | 0.7016 | 0.7405 | 95.5190 / 201.4759 | 60.5659 / 116.4378 | 155.8160 / 314.5104 |
| Vector, no prefix | 0.8619 | 0.9733 | 0.7088 | 0.7430 | 108.0655 / 284.5164 | 79.1665 / 183.1823 | 190.4495 / 434.2187 |
| Hybrid, frozen | 0.8963 | 0.9617 | 0.7233 | 0.7590 | — | — | 144.7082 / 211.0130 |

The prefix did not improve every metric: the no-prefix run was higher on recall@100, MRR@10, and nDCG@10.

| Comparison | Metric | Mean difference | 95% paired interval | Excludes zero |
|---|---|---:|---:|---|
| Hybrid minus vector | nDCG@10 | 0.0186 | [0.0044, 0.0333] | Yes |
| Hybrid minus vector | Recall@10 | 0.0183 | [0.0050, 0.0350] | Yes |
| Hybrid minus vector | MRR@10 | 0.0218 | [0.0040, 0.0404] | Yes |
| Hybrid minus BM25 | nDCG@10 | 0.0968 | [0.0644, 0.1296] | Yes |
| Hybrid minus BM25 | Recall@10 | 0.1121 | [0.0716, 0.1556] | Yes |
| Hybrid minus BM25 | MRR@10 | 0.0931 | [0.0574, 0.1288] | Yes |
| Vector minus BM25 | nDCG@10 | 0.0783 | [0.0401, 0.1155] | Yes |
| Vector minus BM25 | Recall@10 | 0.0937 | [0.0492, 0.1398] | Yes |
| Vector minus BM25 | MRR@10 | 0.0713 | [0.0298, 0.1122] | Yes |

The hybrid-minus-vector interval for each reported metric excludes zero. The hybrid-minus-BM25 interval for each reported metric excludes zero. The vector-minus-BM25 interval for each reported metric excludes zero.

## Failure analysis

Counts below use saved top-10 results for the 300 test queries.

| Category | Count |
|---|---:|
| Hybrid-only hits vs vector | 6 |
| Vector-only hits vs hybrid | 0 |
| Hybrid-only hits vs BM25 | 38 |
| BM25-only hits vs hybrid | 6 |
| Missed by all three | 21 |

| Query ID | Query text | Hybrid rank | Vector rank | BM25 rank |
|---:|---|---:|---:|---:|
| 132 | Aspirin inhibits the production of PGE2. | 31 | 26 | Not in top 100 |
| 239 | Cellular aging closely links to an older appearance. | 33 | 29 | Not in top 100 |
| 421 | Flexible molecules experience greater steric hindrance in the tumor microenviroment than rigid molecules. | Not in top 100 | 51 | Not in top 100 |
| 437 | Functional consequences of genomic alterations due to Myelodysplastic syndrome (MDS) are poorly understood due to the lack of an animal model. | Not in top 100 | Not in top 100 | Not in top 100 |
| 475 | Glycolysis is one of the primary glycometabolic pathways in cells. | 17 | 22 | 12 |
| 502 | Healthcare delivery efficiency in crowded delivery centers is impaired by improving structural, logistical, and interpersonal elements. | Not in top 100 | 85 | Not in top 100 |
| 508 | Hematopoietic Stem Cell purification reaches purity rate of up to 50%. | 17 | 13 | 47 |
| 560 | Immune responses result in the development of inflammatory Th17 cells and anti-inflammatory iTregs. | Not in top 100 | 98 | Not in top 100 |
| 716 | Low expression of miR7a exerts a biological function in testis. | 49 | 30 | Not in top 100 |
| 783 | Mice without IFN-γ or its receptor are resistant to EAM induced with α-MyHC/CFA. | 19 | 21 | 31 |
| 834 | NOX2-independent pathways can generate peroxynitrite by reacting with nitrogen intermediates. | 63 | 88 | 28 |
| 870 | Obesity decreases life quality. | 20 | 18 | Not in top 100 |
| 887 | Only a minority of cells survive development after differentiation into stress-resistant spores. | Not in top 100 | Not in top 100 | Not in top 100 |
| 975 | Primary pro-inflammatory cytokines induce secondary pro- and anti-inflammatory mediators. | Not in top 100 | 71 | Not in top 100 |
| 1110 | Suboptimal nutrition is not predictive of chronic disease | Not in top 100 | 67 | Not in top 100 |
| 1175 | The PPR MDA5 has two N-terminal CARD domains. | Not in top 100 | Not in top 100 | 55 |
| 1199 | The benefits of colchicine were achieved with effective widespread use of secondary prevention strategies such as high-dose statins. | Not in top 100 | Not in top 100 | Not in top 100 |
| 1213 | The deregulated and prolonged activation of monocytes has deleterious effects in inflammatory diseases. | Not in top 100 | Not in top 100 | Not in top 100 |
| 1226 | The loss of the TET protein functions may have dire biological consequences, such as myeloid cancers. | 16 | 14 | 82 |
| 1316 | Transferred UCB T cells acquire a memory-like phenotype in recipients. | 31 | 33 | 26 |
| 1332 | Tumor necrosis factor alpha (TNF-α) and interleukin-1 (IL-1) are pro-inflammatory cytokines that inhibit IL-6 and IL-10. | Not in top 100 | Not in top 100 | Not in top 100 |

## Generation evaluation

The generation task asks a model to return JSON containing a verdict, context document IDs, and a rationale of at most 60 words, using only the supplied abstracts. Prompt version `answer_v1` uses model `gemini-2.5-flash-lite`; the checked-in run has 5 queries and 0 API calls because answers were cached.

| Context | Queries | Validity | Grounding | Citation precision | Citation recall | Failures |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid top-5 | 5 | 5/5 (1.0000) | 5/5 (1.0000) | 1/1 (1.0000) | 1/1 (1.0000) | 0 |
| BM25 top-5 | 5 | 5/5 (1.0000) | 5/5 (1.0000) | 1/1 (1.0000) | 1/1 (1.0000) | 0 |

| Context | No relevant document in top-5: queries / insufficient / rate | Relevant document in top-5: queries / insufficient / rate |
|---|---|---|
| Hybrid | 3 / 3 / 1.0000 | 2 / 1 / 0.5000 |
| BM25 | 3 / 3 / 1.0000 | 2 / 1 / 0.5000 |

Citation precision and recall are averaged only over answers that contain citations. Verdict labels were available in the cached records; verdict accuracy was not evaluated because no independent verdict labels were provided.

Examples, quoted verbatim from `results/generation_eval.md`:

1. “0-dimensional biomaterials show inductive properties.” — “The provided abstracts do not contain information about 0-dimensional biomaterials or their inductive properties.”
2. “1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.” — “The provided abstracts discuss rare variants and their association with diseases or gene expression, but none specifically mention the 1,000 Genomes Project.”
3. “1/2000 in UK have abnormal PrP positivity.” — “A large-scale survey of 32,441 archived appendix samples in the UK found 16 positive for abnormal PrP.”
4. “5% of perinatal mortality is due to low birth weight.” — “The provided abstracts do not contain specific statistics on the percentage of perinatal mortality attributed to low birth weight.”

## Limitations and threats to validity

- Results cover one SciFact version, one local corpus, and one 300-query test split.
- Train tuning and the final test run are separate, but the test set is small.
- Exact search and a single-node database do not establish deployment-scale behavior.
- Generation results are a five-query deterministic smoke evaluation with no human or independent verdict accuracy assessment.


## Reproduction

See [REPRODUCE.md](REPRODUCE.md) for copy-pasteable commands and command verification.
