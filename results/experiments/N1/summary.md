# N1: edge-aware attention

Attention scores get a graph bias: softmax(QKᵀ/√d_h + β[l,h]·B + mask). B is a 24 × 24 z-scored matrix of graph scores for each document's slice, and β is one learnable scalar per layer and head (2 × 4), initialised to 0 and excluded from weight decay.

- **N1a (`intimacy`):** B from log(S + 1e-10), S = intimacy (paper Eq. 2).
- **N1b (`edge`):** B from log(1 + edge weight), TF-IDF for target–helper pairs and positive PMI for helper–helper pairs, 0 without an edge.

Reference: `configs/baseline.yaml`, results reused from `results/baseline/`. Everything except the attention bias is identical (k = 23, intimacy sampler, type and WL encodings on, mean readout, fine-tuned embeddings, same seeds and validation splits).

- **Accuracy and macro-F1:** in %, mean ± sample std (ddof = 1) over seeds 0–4.
- **Δ:** setting mean minus reference mean.
- **p-values:** two-sided paired t-test across the 5 seeds (df = 4); `*` marks p < 0.05. No correction for multiple comparisons.

## R8

| Setting | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | s / epoch |
|---|---|---|---|---|---|---|---|---|
| **reference** | 95.08 ± 0.29 | — | — | 86.50 ± 1.76 | — | — | 5.6 | 3.65 |
| N1a: attn_bias = intimacy | 95.05 ± 0.33 | -0.03 | 0.305 | 86.37 ± 1.86 | -0.13 | 0.156 | 5.6 | 3.48 |
| N1b: attn_bias = edge | 95.07 ± 0.34 | -0.01 | 0.828 | 86.38 ± 1.85 | -0.12 | 0.166 | 5.6 | 3.56 |

Per-seed accuracy (%), seeds 0–4: reference 95.29 95.11 94.61 95.02 95.34; attn_intimacy 95.25 95.11 94.52 94.97 95.39; attn_edge 95.25 95.25 94.52 94.97 95.34

## Ohsumed

| Setting | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | s / epoch |
|---|---|---|---|---|---|---|---|---|
| **reference** | 53.79 ± 1.18 | — | — | 34.74 ± 2.43 | — | — | 5.6 | 2.57 |
| N1a: attn_bias = intimacy | 53.57 ± 1.15 | -0.22 | 0.014* | 34.62 ± 2.35 | -0.12 | 0.088 | 5.6 | 2.58 |
| N1b: attn_bias = edge | 53.70 ± 1.20 | -0.09 | 0.353 | 34.70 ± 2.30 | -0.04 | 0.664 | 5.6 | 2.57 |

Per-seed accuracy (%), seeds 0–4: reference 54.04 53.03 53.55 55.68 52.66; attn_intimacy 53.99 52.71 53.28 55.35 52.51; attn_edge 54.04 52.86 53.18 55.65 52.78

## Learned β at the best checkpoint

Mean ± sample std over the 5 seeds. Rows are layers, columns are heads.

**N1a: attn_bias = intimacy, R8**

| Layer | head 0 | head 1 | head 2 | head 3 |
|---|---|---|---|---|
| 0 | +0.140 ± 0.022 | +0.107 ± 0.031 | +0.116 ± 0.020 | +0.103 ± 0.051 |
| 1 | +0.144 ± 0.035 | +0.122 ± 0.037 | +0.147 ± 0.034 | +0.158 ± 0.037 |

**N1a: attn_bias = intimacy, Ohsumed**

| Layer | head 0 | head 1 | head 2 | head 3 |
|---|---|---|---|---|
| 0 | +0.105 ± 0.026 | +0.096 ± 0.027 | +0.130 ± 0.041 | +0.096 ± 0.021 |
| 1 | +0.171 ± 0.046 | +0.194 ± 0.032 | +0.192 ± 0.026 | +0.179 ± 0.037 |

**N1b: attn_bias = edge, R8**

| Layer | head 0 | head 1 | head 2 | head 3 |
|---|---|---|---|---|
| 0 | +0.120 ± 0.023 | +0.091 ± 0.028 | +0.102 ± 0.032 | +0.090 ± 0.056 |
| 1 | +0.111 ± 0.039 | +0.089 ± 0.029 | +0.113 ± 0.008 | +0.120 ± 0.036 |

**N1b: attn_bias = edge, Ohsumed**

| Layer | head 0 | head 1 | head 2 | head 3 |
|---|---|---|---|---|
| 0 | +0.092 ± 0.025 | +0.088 ± 0.018 | +0.120 ± 0.039 | +0.091 ± 0.018 |
| 1 | +0.146 ± 0.047 | +0.176 ± 0.034 | +0.166 ± 0.028 | +0.158 ± 0.027 |

## Cost of building B (k = 23, intimacy sampler)

| Dataset | Version | Graph rebuild (s) | Raw scores (s) | Normalise (s) | Total (s) | Storage (MB, float32) |
|---|---|---|---|---|---|---|
| R8 | intimacy | 1.4 | 25.3 | 0.10 | 26.8 | 16.9 |
| R8 | edge | 1.3 | 0.3 | 0.11 | 1.8 | 16.9 |
| Ohsumed | intimacy | 3.1 | 70.5 | 0.09 | 73.7 | 16.3 |
| Ohsumed | edge | 3.2 | 0.4 | 0.10 | 3.7 | 16.3 |

Raw intimacy scores need the same dense factorisation of the n × n graph matrix as the main preprocessing. Edge weights are read straight from the sparse graph. B is built once per dataset and cached.

## Target-to-helper attention by intimacy rank (seed 0, test set)

Average attention weight from the target position to one helper in each rank group, averaged over test documents, heads and both layers. For comparison, uniform attention over the 24 positions would be 0.042.

| Dataset | Setting | Self | Ranks 1–5 | Ranks 6–12 | Ranks 13–23 | Seed-0 accuracy reproduced |
|---|---|---|---|---|---|---|
| R8 | reference | 0.042 | 0.043 | 0.042 | 0.041 | yes |
| R8 | attn_intimacy | 0.038 | 0.051 | 0.042 | 0.037 | yes |
| Ohsumed | reference | 0.051 | 0.047 | 0.042 | 0.038 | yes |
| Ohsumed | attn_intimacy | 0.041 | 0.052 | 0.043 | 0.036 | yes |

## Failures

None: all 2 settings × 2 datasets × 5 seeds completed.
