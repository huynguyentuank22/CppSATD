# CppSATD — Self-Admitted Technical Debt in C++ Projects

A multi-class, empirical study of **Self-Admitted Technical Debt (SATD)** in C++ source-code comments, covering three complementary research questions:

| # | Question | Methods |
|---|---|---|
| RQ1 | Do different SATD types carry distinct sentiment signals? | HuggingFace ensemble inference, Cramér's V, χ² |
| RQ2 | Can dense embeddings retrieve same-type SATD comments? | SentenceTransformer kNN, Retrieval@k, intra/inter gap |
| RQ3 | Are embedding-based classifiers trustworthy and explainable? | Logistic Regression, leave-one-token-out occlusion XAI, Jaccard stability & alignment |

---

## Table of Contents

1. [Dataset](#dataset)
2. [Repository Structure](#repository-structure)
3. [Methodology](#methodology)
   - [RQ1 — Sentiment Analysis](#rq1--sentiment-analysis)
   - [RQ2 — Embedding-Based Retrieval](#rq2--embedding-based-retrieval)
   - [RQ3 — Classification & XAI](#rq3--classification--xai)
4. [Results](#results)
   - [RQ1 Results](#rq1-results)
   - [RQ2 Results](#rq2-results)
   - [RQ3 Results](#rq3-results)
5. [Insights](#insights)
6. [Reproducibility](#reproducibility)
7. [Dependencies](#dependencies)

---

## Dataset

**Source:** `manual_annotations_cleaned.csv` — curated manual annotations of C++ source-code comments.

| Attribute | Value |
|---|---|
| Total samples | **35,073** |
| Text column | `commenttext_clean` |
| Label column | `manual_annotation` |

### Class Distribution

| SATD Type | Count | % |
|---|---|---|
| NON-SATD | 22,004 | 62.7 % |
| DESIGN/CODE | 8,242 | 23.5 % |
| REQUIREMENT | 3,917 | 11.2 % |
| DEFECT | 442 | 1.3 % |
| TEST | 422 | 1.2 % |
| DOCUMENTATION | 46 | 0.1 % |

> The dataset is highly imbalanced: NON-SATD is ~5× more frequent than the next class (DESIGN/CODE), and DOCUMENTATION is extremely rare (46 samples).

---

## Repository Structure

```
CppSATD/
├── data/                         # Dataset cards and documentation
│   └── data_and_scripts/
│       ├── cppsatd.csv
│       ├── manual_annotations.csv
│       └── manual_annotations_cleaned.csv
├── notebooks/
│   ├── EDA_CppSATD.ipynb
│   └── preprocessing_CppSATD.ipynb
├── src/
│   ├── rq1_sentiment.py          # RQ1: Sentiment ensemble
│   ├── rq2_embeddings.py         # RQ2: Embeddings + kNN
│   └── rq3_trust_xai.py          # RQ3: Classifier + XAI
└── results/
    ├── rq1/                      # stats.json, crosstab CSVs, plots
    ├── rq2/                      # per-model metrics, embeddings, kNN indices
    └── rq3/                      # classification metrics, XAI, stability, alignment
```

---

## Methodology

### RQ1 — Sentiment Analysis

**Goal:** Test whether SATD type and comment sentiment are statistically dependent.

**Pipeline:**
1. Three pre-trained HuggingFace sentiment models perform inference on all comments.
2. Predictions are combined via soft-voting ensemble (averaged probabilities) into `pos`, `neu`, `neg` labels.
3. A χ² test of independence is run on the 6×3 contingency table (SATD type × sentiment).
4. Effect size is measured with **Cramér's V**.

**Models used:** multilingual/English BERT-based sentiment classifiers from Hugging Face.

---

### RQ2 — Embedding-Based Retrieval

**Goal:** Evaluate whether sentence embeddings form semantically coherent clusters per SATD type.

**Pipeline:**
1. Three SentenceTransformer models encode all comments into L2-normalised vectors.
2. FAISS exact kNN search at k = {10, 50} retrieves nearest neighbours per comment.
3. Two metrics are computed:
   - **Retrieval@k** — fraction of k nearest neighbours sharing the same SATD type.
   - **Intra/Inter cosine gap** — mean(intra-class cosine) − mean(inter-class cosine) per sample.

**Embedding models:**

| Model | Params |
|---|---|
| `paraphrase-MiniLM-L3-v2` | ~17 M |
| `all-MiniLM-L6-v2` | ~23 M |
| `all-MiniLM-L12-v1` | ~33 M |

---

### RQ3 — Classification & XAI

**Goal:** Train classifiers on embeddings ± sentiment and assess explanatory trustworthiness.

**Pipeline:**
1. Stratified 80/20 train-test split (fixed seed).
2. Two Logistic Regression models (multinomial, L-BFGS, C=1.0):
   - **Model B** — embedding features only.
   - **Model C** — embedding + ensemble sentiment features (concatenation).
3. **XAI (leave-one-token-out occlusion):** For each sample a custom regex tokeniser produces candidate tokens; each token is masked once and the change in predicted-class probability is the importance score. Top-10 tokens are recorded.
4. **Stability** — XAI is re-run on a perturbed copy (lowercasing, whitespace normalisation, digit/URL replacement) and Jaccard similarity of top-10 token sets is computed.
5. **Alignment** — Jaccard similarity between a sample's top-10 tokens and those of its 10 nearest embedding neighbours.

---

## Results

### RQ1 Results

#### Statistical Test

| Metric | Value |
|---|---|
| χ² statistic | 979.46 |
| p-value | 4.96 × 10⁻²⁰⁴ |
| Degrees of freedom | 10 |
| Cramér's V | **0.1182** |
| N | 35,073 |
| Full model agreement | 87.69 % |
| Majority agreement | 99.97 % |

The association between SATD type and ensemble sentiment is **highly significant** (p ≪ 0.001) but the effect size (Cramér's V = 0.12) is **small**. Sentiment alone cannot distinguish SATD types, but systematic differences do exist.

#### Ensemble Sentiment Distribution

| SATD Type | Positive | Neutral | Negative |
|---|---|---|---|
| NON-SATD | 0.21 % | 93.2 % | 6.5 % |
| DESIGN/CODE | 0.33 % | 87.7 % | 12.0 % |
| REQUIREMENT | 0.15 % | 94.97 % | 4.9 % |
| TEST | 0.24 % | 92.7 % | 7.1 % |
| **DEFECT** | 0.00 % | 58.1 % | **41.9 %** |
| DOCUMENTATION | 0.00 % | 97.8 % | 2.2 % |

> **Key finding:** DEFECT comments are overwhelmingly predicted as negative (41.9 %), far above any other type. DESIGN/CODE also attracts more negative sentiment (12 %) than NON-SATD or DOCUMENTATION, consistent with the frustration often embedded in debt acknowledgements.

#### Ensemble Label Distribution

| Label | Count |
|---|---|
| Neutral | 32,159 (91.7 %) |
| Negative | 2,833 (8.1 %) |
| Positive | 81 (0.2 %) |

---

### RQ2 Results

#### Retrieval@k — Overall

| Model | Retrieval@10 | Retrieval@50 |
|---|---|---|
| **paraphrase-MiniLM-L3-v2** | **72.1 %** | **66.9 %** |
| all-MiniLM-L6-v2 | 65.0 % | 60.2 % |
| all-MiniLM-L12-v1 | 65.2 % | 60.1 % |

`paraphrase-MiniLM-L3-v2`, despite being the smallest model, consistently outperforms the others on retrieval.

#### Retrieval@10 — Per SATD Type (best model)

| SATD Type | Retrieval@10 |
|---|---|
| NON-SATD | 85.5 % |
| DESIGN/CODE | 52.0 % |
| REQUIREMENT | 49.0 % |
| TEST | 42.2 % |
| DEFECT | 21.2 % |
| DOCUMENTATION | 4.8 % |

#### Intra/Inter Cosine Gap@10 — Per SATD Type (paraphrase-MiniLM-L3-v2)

| SATD Type | Gap mean |
|---|---|
| DOCUMENTATION | +0.270 |
| DEFECT | +0.149 |
| TEST | +0.083 |
| REQUIREMENT | +0.047 |
| DESIGN/CODE | +0.040 |
| NON-SATD | +0.019 |

> **Key finding:** Despite poor retrieval, DOCUMENTATION and DEFECT have the largest intra/inter cosine gaps — their embeddings form internally tight clusters, but those clusters are very small and under‑represented. NON-SATD has near-zero gap, suggesting its vast majority drowns out within-class structure. Overall gap mean remains low (≈ 0.03), indicating the embedding space is not strongly partitioned by SATD type.

---

### RQ3 Results

#### Classification Performance

| Model | Embedder | Macro-F1 | Balanced Accuracy |
|---|---|---|---|
| B (emb only) | paraphrase-MiniLM-L3-v2 | 0.4927 | 0.4465 |
| C (emb + sent) | paraphrase-MiniLM-L3-v2 | 0.4973 | 0.4519 |
| B (emb only) | all-MiniLM-L6-v2 | 0.4857 | 0.4431 |
| C (emb + sent) | all-MiniLM-L6-v2 | 0.4958 | 0.4522 |
| B (emb only) | all-MiniLM-L12-v1 | 0.4986 | 0.4576 |
| **C (emb + sent)** | **all-MiniLM-L12-v1** | **0.5127** | **0.4699** |

> Adding sentiment features (Model C) uniformly lifts Macro-F1 by ~0.01–0.014 points. The largest model (L12-v1 + sentiment) achieves the best Macro-F1 of 0.513. All Macro-F1 values cluster near 0.50 with Balanced Accuracy near 0.45, reflecting the difficulty of this imbalanced 6-class problem.

#### XAI Stability (mean Jaccard of top-10 tokens after perturbation)

| Embedder | Model B | Model C |
|---|---|---|
| paraphrase-MiniLM-L3-v2 | 0.538 | 0.539 |
| all-MiniLM-L6-v2 | 0.508 | 0.509 |
| all-MiniLM-L12-v1 | 0.495 | 0.495 |

**Per-type stability (best model — paraphrase-MiniLM-L3-v2, Model B):**

| SATD Type | Stability |
|---|---|
| NON-SATD | 0.624 |
| DESIGN/CODE | 0.522 |
| REQUIREMENT | 0.500 |
| DEFECT | 0.477 |
| TEST | 0.426 |
| DOCUMENTATION | 0.418 |

> **Key finding:** Explanations are moderately stable (Jaccard ≈ 0.50–0.54). NON-SATD explanations are the most reproducible under surface perturbations; DOCUMENTATION and TEST explanations are the least stable, likely due to short, keyword-heavy comments where a single perturbation changes the entire token context.

#### XAI Alignment (mean Jaccard vs 10 nearest embedding neighbours)

| Embedder | Model B | Model C |
|---|---|---|
| paraphrase-MiniLM-L3-v2 | 0.183 | 0.183 |
| all-MiniLM-L6-v2 | 0.164 | 0.164 |
| all-MiniLM-L12-v1 | 0.165 | 0.165 |

> Alignment scores are low (≈ 0.16–0.18), meaning that even semantically similar comments (per embedding distance) do not share the same explanatory tokens. This reflects the lexical diversity within SATD types — comments can be semantically close yet use completely different vocabulary.

#### Spearman ρ (cosine similarity vs Jaccard alignment)

| Embedder | ρ | p-value |
|---|---|---|
| paraphrase-MiniLM-L3-v2 | 0.050 | 9.7 × 10⁻¹¹ |
| all-MiniLM-L6-v2 | 0.040 | 1.8 × 10⁻⁷ |
| all-MiniLM-L12-v1 | 0.037 | 1.3 × 10⁻⁶ |

> Statistically significant but near-zero correlation — embedding proximity only very weakly predicts explanation similarity. The geometric and the explanatory views of similarity are largely decoupled.

---

## Insights

### 1. Sentiment is a weak but non-trivial signal
The χ² test is overwhelming (p ≈ 10⁻²⁰⁴), yet Cramér's V = 0.12 shows the practical effect is modest. The dominant pattern is **DEFECT ↔ negative sentiment** (41.9 % negative). For classification, sentiment adds ~0.01–0.014 Macro-F1, a small but consistent improvement. Sentiment features cannot replace richer representations, but they act as a useful complementary signal.

### 2. NON-SATD monopolises the embedding space
With 62.7 % of samples, NON-SATD achieves 85.5 % Retrieval@10 simply because its sheer volume dominates the neighbourhood structure. Rare classes (DOCUMENTATION: 0.1 %, DEFECT: 1.3 %) suffer retrieval rates of only 4.8 % and 21.2 % respectively. This class imbalance is the primary bottleneck — not the embedding quality itself (DOCUMENTATION has the highest cosine gap despite the worst retrieval).

### 3. Smaller ≠ worse for paraphrase tasks
`paraphrase-MiniLM-L3-v2` (17 M params) outperforms both larger models on retrieval across all k and all types. This supports the hypothesis that SATD comments resemble paraphrase pairs (same intent expressed differently) more closely than they resemble general semantic similarity tasks. Model architecture matters as much as scale for domain-specific retrieval.

### 4. Classification difficulty reflects inherent label ambiguity
Macro-F1 ≈ 0.50 and Balanced Accuracy ≈ 0.45–0.47 are consistent across all three embedding models and both feature sets. The ceiling effect — where adding a larger model or sentiment features produces only marginal gains — suggests that the bottleneck is **label ambiguity** (comments can plausibly belong to multiple SATD types) rather than feature expressiveness.

### 5. XAI explanations are moderately stable but locally inconsistent
Stability (Jaccard ≈ 0.50–0.54) means that roughly half the top-10 tokens survive surface perturbations — acceptable for practical use but far from robust. The low alignment (≈ 0.16–0.18) and near-zero Spearman ρ (0.04–0.05) reveal that **embedding closeness does not imply explanatory closeness**. Two comments can live in the same neighbourhood in latent space while depending on entirely different keywords for their classification, which poses a challenge for human-in-the-loop SATD review workflows.

### 6. Rare classes are doubly disadvantaged
DOCUMENTATION and DEFECT suffer on every axis: low retrieval, low stability, low alignment. Active learning or data augmentation strategies specifically targeting these classes are warranted before deploying any downstream tool.

---

## Reproducibility

All experiments are fully self-contained in the three `src/` scripts. Each script writes cached artefacts (embeddings, kNN indices, splits, XAI results) and skips recomputation unless `FORCE_RERUN = True`.

### Run order (Kaggle / local)

```bash
# 1. Sentiment inference & stats
python src/rq1_sentiment.py

# 2. Embeddings + kNN
python src/rq2_embeddings.py

# 3. Classifier + XAI (depends on RQ1 and RQ2 outputs)
python src/rq3_trust_xai.py
```

### Key config variables (top of each file)

| Variable | Purpose |
|---|---|
| `DATA_PATH` | Path to `manual_annotations_cleaned.csv` |
| `OUTPUT_DIR` | Root directory for all outputs |
| `RQ1_DIR` | Folder containing `sentiment_predictions.parquet` (RQ3 only) |
| `RQ2_DIR` | Folder containing `embeddings/`, `knn/`, `label_mapping.json` (RQ3 only) |
| `FORCE_RERUN` | Set to `True` to bypass all caches |
| `SEED` | Global random seed (default 42) |

---

## Dependencies

```
torch
sentence-transformers
transformers
faiss-cpu          # or faiss-gpu
scikit-learn
pandas
numpy
scipy
matplotlib
tqdm
pyarrow            # for parquet I/O
```

Install with:

```bash
pip install torch sentence-transformers transformers faiss-cpu \
            scikit-learn pandas numpy scipy matplotlib tqdm pyarrow
```
