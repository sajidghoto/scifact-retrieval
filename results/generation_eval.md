# Generation evaluation

- Date: 2026-11-10
- Prompt version: `answer_v1`
- Model: `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c`
- API calls made: 0

## hybrid

```json
{
  "queries": 300,
  "validity_rate": {
    "count": 299,
    "denominator": 300,
    "value": 0.9966666666666667
  },
  "grounding_rate": {
    "count": 299,
    "denominator": 300,
    "value": 0.9966666666666667
  },
  "citation_precision": {
    "count": 256,
    "denominator": 256,
    "value": 0.6870442708333334
  },
  "citation_recall": {
    "count": 256,
    "denominator": 256,
    "value": 0.7744791666666667
  },
  "answered_with_relevant_citation_rate": {
    "count": 205,
    "denominator": 256,
    "value": 0.80078125
  },
  "abstention": {
    "no_relevant_in_top5": {
      "queries": 49,
      "insufficient": 16,
      "abstention_rate": 0.32653061224489793
    },
    "relevant_in_top5": {
      "queries": 251,
      "insufficient": 27,
      "abstention_rate": 0.10756972111553785
    }
  },
  "failures": 1
}
```

## bm25

```json
{
  "queries": 300,
  "validity_rate": {
    "count": 299,
    "denominator": 300,
    "value": 0.9966666666666667
  },
  "grounding_rate": {
    "count": 299,
    "denominator": 300,
    "value": 0.9966666666666667
  },
  "citation_precision": {
    "count": 245,
    "denominator": 245,
    "value": 0.6893197278911566
  },
  "citation_recall": {
    "count": 245,
    "denominator": 245,
    "value": 0.7406122448979592
  },
  "answered_with_relevant_citation_rate": {
    "count": 189,
    "denominator": 245,
    "value": 0.7714285714285715
  },
  "abstention": {
    "no_relevant_in_top5": {
      "queries": 76,
      "insufficient": 34,
      "abstention_rate": 0.4473684210526316
    },
    "relevant_in_top5": {
      "queries": 224,
      "insufficient": 20,
      "abstention_rate": 0.08928571428571429
    }
  },
  "failures": 1
}
```

## Examples

1. **grounded / hybrid** — 0-dimensional biomaterials show inductive properties.  
Cited IDs: `[]`  
Rationale: No document addresses 0-dimensional biomaterials or their inductive properties directly.
2. **grounded / hybrid** — 1,000 genomes project enables mapping of genetic sequence variation consisting of rare variants with larger penetrance effects than common variants.  
Cited IDs: `['4414547', '14717500']`  
Rationale: The 1,000 Genomes Project supports rare variant analysis, as shown by PPM1D mutations and synthetic associations in the context.
3. **grounded / hybrid** — 1/2000 in UK have abnormal PrP positivity.  
Cited IDs: `['13734012']`  
Rationale: The study found 493 per million prevalence of abnormal PrP, contradicting the 1/2000 (500 per million) claim.
4. **grounded / hybrid** — 5% of perinatal mortality is due to low birth weight.  
Cited IDs: `[]`  
Rationale: No document directly addresses the 5% perinatal mortality due to low birth weight claim.
5. **failure / hybrid** — A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.  
Cited IDs: `[]`  
Rationale: No document directly addresses the specific claim about 1,000 asymptomatic vCJD carriers in the UK.
6. **failure / hybrid** — Active H. pylori urease has a polymeric structure that compromises two subunits, UreA and UreB.  
Cited IDs: `[]`  
Rationale: No document directly addresses the structure of H. pylori urease subunits.
7. **failure / hybrid** — Alizarin forms hydrogen bonds with residues involved in PGAM1 substrate binding.  
Cited IDs: `[]`  
Rationale: No document mentions alizarin or PGAM1 hydrogen bonding interactions.
8. **failure / hybrid** — CCL19 is absent within dLNs.  
Cited IDs: `[]`  
Rationale: No document explicitly states whether CCL19 is absent in dLNs.
