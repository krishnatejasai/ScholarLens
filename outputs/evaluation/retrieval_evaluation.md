# ScholarLens Retrieval Evaluation

## Evaluation Setup

- Evaluation queries: 60
- Corpus papers: 20
- Indexed chunks: 2179
- Dense retrieval depth: 50
- Reranking candidate depth: 30
- Evaluation level: unique paper ranking

## Aggregate Results

| Metric | Dense Retrieval | Dense + Reranking | Relative Change |
|---|---:|---:|---:|
| Recall@1 | 0.7500 | 0.7917 | +5.56% |
| Recall@3 | 0.9583 | 0.9500 | -0.87% |
| Recall@5 | 0.9833 | 0.9833 | +0.00% |
| MRR | 0.9028 | 0.9292 | +2.92% |
| nDCG@5 | 0.9183 | 0.9365 | +1.98% |

## Per-Query Results

| Query | Dense MRR | Reranked MRR | Dense nDCG@5 | Reranked nDCG@5 |
|---|---:|---:|---:|---:|
| q01 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q02 | 1.0000 | 0.5000 | 1.0000 | 0.6309 |
| q03 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q04 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q05 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q06 | 0.3333 | 0.5000 | 0.5000 | 0.6309 |
| q07 | 0.3333 | 0.5000 | 0.5000 | 0.6309 |
| q08 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q09 | 1.0000 | 1.0000 | 0.9738 | 0.9828 |
| q10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q15 | 1.0000 | 0.5000 | 1.0000 | 0.4777 |
| q16 | 1.0000 | 1.0000 | 0.9173 | 0.9173 |
| q17 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q21 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q23 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q24 | 0.5000 | 1.0000 | 0.6309 | 1.0000 |
| q25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q26 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q27 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q28 | 0.5000 | 1.0000 | 0.6309 | 1.0000 |
| q29 | 0.5000 | 1.0000 | 0.6309 | 1.0000 |
| q30 | 0.3333 | 1.0000 | 0.5000 | 1.0000 |
| q31 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q32 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q33 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q34 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q35 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q36 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q37 | 1.0000 | 1.0000 | 0.7098 | 0.7098 |
| q38 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q39 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q40 | 1.0000 | 1.0000 | 0.9828 | 1.0000 |
| q41 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q42 | 1.0000 | 1.0000 | 0.9738 | 0.9828 |
| q43 | 0.3333 | 0.2500 | 0.5000 | 0.4307 |
| q44 | 0.3333 | 0.5000 | 0.5000 | 0.6309 |
| q45 | 0.5000 | 1.0000 | 0.6352 | 1.0000 |
| q46 | 0.5000 | 1.0000 | 0.6309 | 1.0000 |
| q47 | 1.0000 | 0.5000 | 0.9828 | 0.6443 |
| q48 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q49 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q50 | 1.0000 | 0.5000 | 1.0000 | 0.6309 |
| q51 | 1.0000 | 1.0000 | 0.9828 | 0.9738 |
| q52 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q53 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q54 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q55 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q56 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q57 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q58 | 1.0000 | 1.0000 | 0.9173 | 0.9173 |
| q59 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| q60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Interpretation

The benchmark compares dense semantic retrieval against a two-stage pipeline that applies cross-encoder reranking to dense retrieval candidates.

Recall measures whether labeled relevant papers are present within the top-ranked results. MRR measures how early the first relevant paper appears, while nDCG@5 rewards rankings that place highly relevant papers above partially relevant supporting papers.

## Evaluation Limitations

- Relevance judgments are manually defined at the paper level rather than the individual evidence-chunk level.
- The benchmark contains a focused collection of research papers and should not be interpreted as a general web-scale retrieval benchmark.
- Queries are designed around the technical themes represented in the evaluation corpus.