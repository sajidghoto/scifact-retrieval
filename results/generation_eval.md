# Generation evaluation

- Date: 2026-11-03
- Prompt version: `answer_v1`
- Model: `gemini-2.5-flash-lite`
- API calls made: 0

## hybrid

```json
{
  "queries": 5,
  "validity_rate": {
    "count": 5,
    "denominator": 5,
    "value": 1.0
  },
  "grounding_rate": {
    "count": 5,
    "denominator": 5,
    "value": 1.0
  },
  "citation_precision": {
    "count": 1,
    "denominator": 1,
    "value": 1.0
  },
  "citation_recall": {
    "count": 1,
    "denominator": 1,
    "value": 1.0
  },
  "answered_with_relevant_citation_rate": {
    "count": 1,
    "denominator": 1,
    "value": 1.0
  },
  "abstention": {
    "no_relevant_in_top5": {
      "queries": 3,
      "insufficient": 3,
      "abstention_rate": 1.0
    },
    "relevant_in_top5": {
      "queries": 2,
      "insufficient": 1,
      "abstention_rate": 0.5
    }
  },
  "failures": 0
}
```

## bm25

```json
{
  "queries": 5,
  "validity_rate": {
    "count": 5,
    "denominator": 5,
    "value": 1.0
  },
  "grounding_rate": {
    "count": 5,
    "denominator": 5,
    "value": 1.0
  },
  "citation_precision": {
    "count": 1,
    "denominator": 1,
    "value": 1.0
  },
  "citation_recall": {
    "count": 1,
    "denominator": 1,
    "value": 1.0
  },
  "answered_with_relevant_citation_rate": {
    "count": 1,
    "denominator": 1,
    "value": 1.0
  },
  "abstention": {
    "no_relevant_in_top5": {
      "queries": 3,
      "insufficient": 3,
      "abstention_rate": 1.0
    },
    "relevant_in_top5": {
      "queries": 2,
      "insufficient": 1,
      "abstention_rate": 0.5
    }
  },
  "failures": 0
}
```

## Examples

1. **grounded / hybrid** — 0-dimensional biomaterials show inductive properties.  
Cited IDs: `[]`  
Rationale: The provided abstracts do not contain information about 0-dimensional biomaterials or their inductive properties. The documents discuss bone fracture, cytotoxicity assays, surgical meshes, and graphene-based platforms for stem cell culture.
2. **grounded / hybrid** — 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.  
Cited IDs: `[]`  
Rationale: The provided abstracts discuss rare variants and their association with diseases or gene expression, but none specifically mention the 1,000 Genomes Project or compare the penetrance effects of rare versus common variants in the context of that project.
3. **grounded / hybrid** — 1/2000 in UK have abnormal PrP positivity.  
Cited IDs: `['13734012']`  
Rationale: A large-scale survey of 32,441 archived appendix samples in the UK found 16 positive for abnormal PrP, indicating an overall prevalence of 493 per million population, which is approximately 1 in 2028.
4. **grounded / hybrid** — 5% of perinatal mortality is due to low birth weight.  
Cited IDs: `[]`  
Rationale: The provided abstracts do not contain specific statistics on the percentage of perinatal mortality attributed to low birth weight. While low birth weight is mentioned as a factor in some contexts, its precise contribution to perinatal mortality is not quantified.
5. **failure / hybrid** — 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.  
Cited IDs: `[]`  
Rationale: The provided abstracts discuss rare variants and their association with diseases or gene expression, but none specifically mention the 1,000 Genomes Project or compare the penetrance effects of rare versus common variants in the context of that project.
6. **failure / bm25** — 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.  
Cited IDs: `[]`  
Rationale: The provided abstracts do not contain information about the 1,000 genomes project specifically enabling the mapping of genetic sequence variation or comparing the penetrance effects of rare versus common variants.
