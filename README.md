# CppSATD — Self-Admitted Technical Debt in C++ Projects

> All metrics are **macro-averaged** on the test set (80/10/10 stratified split, seed=42).  
> Each row shows the best-performing LR for that encoder/combo, ranked by macro F1.

---

## Phase 1 — Comment-only Baselines

| Rank | comment_encoder | Accuracy | Precision | Recall | F1 |
|------|----------------|----------|-----------|--------|----|
| 🥇 1 | `microsoft/deberta-base` | 0.9385 | 0.8033 | 0.7626 | **0.7771** |
| 🥈 2 | `roberta-base` | 0.9288 | 0.8436 | 0.7364 | **0.7445** |
| 🥉 3 | `bert-base-uncased` | 0.9288 | 0.8576 | 0.6902 | **0.7271** |

## Phase 2 — Code-only Baselines

| Rank | code_encoder | Accuracy | Precision | Recall | F1 |
|------|-------------|----------|-----------|--------|----|
| 🥇 1 | `microsoft/unixcoder-base-nine` | 0.6869 | 0.5298 | 0.4660 | **0.4863** |
| 🥈 2 | `microsoft/graphcodebert-base` | 0.6444 | 0.4309 | 0.4810 | **0.4495** |
| 🥉 3 | `microsoft/codebert-base` | 0.6362 | 0.4302 | 0.4776 | **0.4440** |

## Phase 3 — Late-Fusion Dual Encoder

| Rank | comment_encoder | code_encoder | Accuracy | Precision | Recall | F1 |
|------|----------------|--------------|----------|-----------|--------|----|
| 🥇 1 | `microsoft/deberta-base` | `microsoft/graphcodebert-base` | 0.9373 | 0.7639 | 0.7687 | **0.7640** |
| 🥈 2 | `microsoft/deberta-base` | `microsoft/unixcoder-base-nine` | 0.9345 | 0.7488 | 0.7649 | **0.7530** |
| 🥉 3 | `roberta-base` | `microsoft/unixcoder-base-nine` | 0.9325 | 0.8457 | 0.7335 | **0.7445** |
| 4 | `roberta-base` | `microsoft/graphcodebert-base` | 0.9319 | 0.7724 | 0.7400 | **0.7438** |
| 5 | `bert-base-uncased` | `microsoft/unixcoder-base-nine` | 0.9299 | 0.7501 | 0.7240 | **0.7252** |
| 6 | `microsoft/deberta-base` | `microsoft/codebert-base` | 0.9348 | 0.7161 | 0.7329 | **0.7222** |
| 7 | `bert-base-uncased` | `microsoft/codebert-base` | 0.9379 | 0.6935 | 0.6955 | **0.6944** |
| 8 | `bert-base-uncased` | `microsoft/graphcodebert-base` | 0.9370 | 0.6801 | 0.6939 | **0.6861** |
| 9 | `roberta-base` | `microsoft/codebert-base` | 0.9282 | 0.6780 | 0.6869 | **0.6821** |

## Phase 4 — Cross-Attention Dual Encoder

| Rank | comment_encoder | code_encoder | Accuracy | Precision | Recall | F1 |
|------|----------------|--------------|----------|-----------|--------|----|
| 🥇 1 | `microsoft/deberta-base` | `microsoft/unixcoder-base-nine` | 0.9376 | 0.8184 | 0.7610 | **0.7815** |
| 🥈 2 | `roberta-base` | `microsoft/unixcoder-base-nine` | 0.9356 | 0.8431 | 0.7461 | **0.7483** |
| 🥉 3 | `microsoft/deberta-base` | `microsoft/graphcodebert-base` | 0.9336 | 0.7790 | 0.7239 | **0.7400** |
