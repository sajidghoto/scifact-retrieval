# Paired bootstrap test

Per-query metrics were recomputed from saved top-10 document IDs using the existing metrics module.
Paired bootstrap: 10,000 resamples, seed `20261103`.

| Comparison | Metric | Mean difference | 95% percentile interval | First better share | Interval excludes zero |
|---|---|---:|---:|---:|---|
| hybrid vs vector | ndcg@10 | 0.018580 | [0.004365, 0.033348] | 0.996500 | yes |
| hybrid vs vector | recall@10 | 0.018333 | [0.005000, 0.035000] | 0.994800 | yes |
| hybrid vs vector | mrr@10 | 0.021779 | [0.003964, 0.040376] | 0.993200 | yes |
| hybrid vs bm25 | ndcg@10 | 0.096836 | [0.064441, 0.129596] | 1.000000 | yes |
| hybrid vs bm25 | recall@10 | 0.112056 | [0.071554, 0.155612] | 1.000000 | yes |
| hybrid vs bm25 | mrr@10 | 0.093101 | [0.057432, 0.128824] | 1.000000 | yes |
| vector vs bm25 | ndcg@10 | 0.078256 | [0.040097, 0.115476] | 1.000000 | yes |
| vector vs bm25 | recall@10 | 0.093722 | [0.049164, 0.139778] | 0.999900 | yes |
| vector vs bm25 | mrr@10 | 0.071321 | [0.029837, 0.112170] | 0.999800 | yes |
