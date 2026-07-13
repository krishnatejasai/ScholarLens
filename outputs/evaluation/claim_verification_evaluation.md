# ScholarLens Claim Verification Evaluation

## Evaluation Setup

- Total claims: 60
- Development claims: 30
- Held-out test claims: 30
- Balanced labels: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE
- Evidence source: dense retrieval followed by cross-encoder reranking
- Verification model: cross-encoder/nli-deberta-v3-base

## Aggregate Results

| Split | Claims | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Overall | 60 | 0.7500 | 0.7489 |
| Development | 30 | 0.7667 | 0.7662 |
| Held-out test | 30 | 0.7333 | 0.7262 |

## Overall Per-Class Results

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUPPORTED | 1.0000 | 0.6000 | 0.7500 | 20 |
| CONTRADICTED | 0.6333 | 0.9500 | 0.7600 | 20 |
| INSUFFICIENT_EVIDENCE | 0.7778 | 0.7000 | 0.7368 | 20 |

### Overall Confusion Matrix

| True / Predicted | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |
|---|---:|---:|---:|
| SUPPORTED | 12 | 5 | 3 |
| CONTRADICTED | 0 | 19 | 1 |
| INSUFFICIENT_EVIDENCE | 0 | 6 | 14 |

## Development Per-Class Results

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUPPORTED | 1.0000 | 0.7000 | 0.8235 | 10 |
| CONTRADICTED | 0.6250 | 1.0000 | 0.7692 | 10 |
| INSUFFICIENT_EVIDENCE | 0.8571 | 0.6000 | 0.7059 | 10 |

### Development Confusion Matrix

| True / Predicted | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |
|---|---:|---:|---:|
| SUPPORTED | 7 | 2 | 1 |
| CONTRADICTED | 0 | 10 | 0 |
| INSUFFICIENT_EVIDENCE | 0 | 4 | 6 |

## Held-Out Test Per-Class Results

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| SUPPORTED | 1.0000 | 0.5000 | 0.6667 | 10 |
| CONTRADICTED | 0.6429 | 0.9000 | 0.7500 | 10 |
| INSUFFICIENT_EVIDENCE | 0.7273 | 0.8000 | 0.7619 | 10 |

### Held-Out Test Confusion Matrix

| True / Predicted | SUPPORTED | CONTRADICTED | INSUFFICIENT_EVIDENCE |
|---|---:|---:|---:|
| SUPPORTED | 5 | 3 | 2 |
| CONTRADICTED | 0 | 9 | 1 |
| INSUFFICIENT_EVIDENCE | 0 | 2 | 8 |

## Interpretation

ScholarLens performs three-way evidence-based claim verification after dense retrieval and cross-encoder reranking. Development results may be used for system analysis, while held-out test results represent the primary final benchmark.

## Limitations

- The benchmark contains manually authored claims derived from a focused 20-paper corpus.
- Verification performance depends jointly on evidence retrieval, reranking, and NLI classification.
- Full-passage NLI may be affected by irrelevant text, bibliographic content, or multiple claims within a chunk.
- The results should not be interpreted as performance on general open-domain scientific fact verification.