# Retrieval evaluation (small, fictional set)

12 fictional notes, 13 hand-written queries with hand-labelled relevant notes. Metrics are macro-averaged over queries; MRR uses the top 10.

## Embedder: hashing - all queries (n=13)

| Retriever | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| vector only | 0.814 | 0.897 | 0.859 |
| graph only | 0.840 | 1.000 | 0.833 |
| hybrid (1 hop) | 0.872 | 0.974 | 0.821 |
| hybrid (2 hops) | 0.872 | 0.955 | 0.859 |

## Embedder: hashing - lexical queries (n=5)

| Retriever | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| vector only | 1.000 | 1.000 | 0.867 |
| graph only | 1.000 | 1.000 | 0.867 |
| hybrid (1 hop) | 1.000 | 1.000 | 0.867 |
| hybrid (2 hops) | 1.000 | 1.000 | 0.867 |

## Embedder: hashing - relational queries (n=8)

| Retriever | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| vector only | 0.698 | 0.833 | 0.854 |
| graph only | 0.740 | 1.000 | 0.812 |
| hybrid (1 hop) | 0.792 | 0.958 | 0.792 |
| hybrid (2 hops) | 0.792 | 0.927 | 0.854 |

## Embedder: minilm - all queries (n=13)

| Retriever | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| vector only | 0.731 | 0.731 | 0.782 |
| graph only | 0.756 | 0.878 | 0.821 |
| hybrid (1 hop) | 0.712 | 0.782 | 0.821 |
| hybrid (2 hops) | 0.769 | 0.878 | 0.821 |

## Embedder: minilm - lexical queries (n=5)

| Retriever | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| vector only | 1.000 | 1.000 | 0.900 |
| graph only | 1.000 | 1.000 | 0.867 |
| hybrid (1 hop) | 1.000 | 1.000 | 0.900 |
| hybrid (2 hops) | 1.000 | 1.000 | 0.900 |

## Embedder: minilm - relational queries (n=8)

| Retriever | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| vector only | 0.562 | 0.562 | 0.708 |
| graph only | 0.604 | 0.802 | 0.792 |
| hybrid (1 hop) | 0.531 | 0.646 | 0.771 |
| hybrid (2 hops) | 0.625 | 0.802 | 0.771 |
