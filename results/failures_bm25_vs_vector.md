# BM25 vs. Vector Failure Analysis

These counts use the measured top-10 results for the 300 qrel-backed test
queries. Vector examples use the query-prefix variant.

| Category | Count |
|---|---:|
| BM25-only hits | 12 |
| Vector-only hits | 38 |
| Both hit | 229 |
| Both miss | 21 |

## BM25-only hits

| Query ID | Query | Relevant document IDs | BM25 top-10 includes | Vector-prefix top-5 |
|---|---|---|---|---|
| 127 | Arginine 90 in p150n is important for interaction with EB1. | 21598000 | 21598000 | 11181416, 22937651, 10423989, 35231675, 38623601 |
| 128 | Arterioles have a larger lumen diameter than venules. | 8290953 | 8290953 | 23513718, 2260571, 12549585, 79447, 19854543 |
| 303 | DMRT1 is a sex-determining gene that is epigenetically regulated by the MHM region. | 4388470 | 4388470 | 4085204, 28390999, 23269537, 2086909, 5278233 |
| 312 | De novo assembly of sequence data has more specific contigs than unassembled sequence data. | 6173523 | 6173523 | 14464451, 5650232, 45200347, 14652521, 14915566 |
| 619 | Increased vessel density along with a reduction in fibrosis decreases the efficacy of chemotherapy treatments. | 20888849, 2565138 | 20888849 | 6334188, 25175997, 1171121, 25589047, 5254741 |

## Vector-only hits

| Query ID | Query | Relevant document IDs | Vector-prefix top-10 includes |
|---|---|---|---|
| 1 | 0-dimensional biomaterials show inductive properties. | 31715818 | 31715818 |
| 13 | 5% of perinatal mortality is due to low birth weight. | 1606628 | 1606628 |
| 54 | AMP-activated protein kinase (AMPK) activation increases inflammation-related fibrosis in the lungs. | 49556906 | 49556906 |
| 238 | Cells undergoing methionine restriction may activate miRNAs. | 2251426 | 2251426 |
| 384 | Epidemiological disease burden from noncommunicable diseases is more prevalent in low economic settings. | 13770184 | 13770184 |

## Both miss

There were 21 queries for which neither measured top-10 ranking contained a
relevant document.
