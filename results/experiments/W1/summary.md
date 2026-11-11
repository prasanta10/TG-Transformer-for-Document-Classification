# W1: WL computed inside each sampled slice

The baseline computes WL codes once on the full text graph. W1 follows the paper's wording more literally: WL refinement runs inside each document's 24-node slice (the induced subgraph, unweighted, no self-loops), with one code dictionary shared by all slices. Only the WL codes change; the Eq. 5 encoding and everything else are as in the baseline.

Reference: `configs/baseline.yaml`, results reused from `results/baseline/`. The "WL off" row is E1 "type on, WL off", reused from `results/experiments/e1/type_on_wl_off/`.

- **Accuracy and macro-F1:** in %, mean ± sample std (ddof = 1) over seeds 0–4.
- **Δ:** setting mean minus baseline mean.
- **p-values:** two-sided paired t-test across the 5 seeds (df = 4); `*` marks p < 0.05. No correction for multiple comparisons.

## R8

| Setting | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | s / epoch |
|---|---|---|---|---|---|---|---|---|
| **baseline (full-graph WL)** | 95.08 ± 0.29 | — | — | 86.50 ± 1.76 | — | — | 5.6 | 3.65 |
| W1: subgraph WL | 94.87 ± 0.32 | -0.20 | 0.327 | 84.21 ± 1.03 | -2.29 | 0.100 | 5.2 | 3.38 |
| E1: type on, WL off | 94.98 ± 0.41 | -0.09 | 0.483 | 86.15 ± 0.89 | -0.35 | 0.671 | 5.4 | 3.32 |

Per-seed accuracy (%), seeds 0–4: baseline 95.29 95.11 94.61 95.02 95.34; w1 94.43 95.07 94.66 95.16 95.07; nowl 94.97 95.20 94.29 95.29 95.16

## Ohsumed

| Setting | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | s / epoch |
|---|---|---|---|---|---|---|---|---|
| **baseline (full-graph WL)** | 53.79 ± 1.18 | — | — | 34.74 ± 2.43 | — | — | 5.6 | 2.57 |
| W1: subgraph WL | 51.49 ± 1.71 | -2.31 | 0.010* | 31.58 ± 2.70 | -3.16 | 0.002* | 5.6 | 2.49 |
| E1: type on, WL off | 50.13 ± 2.43 | -3.66 | 0.016* | 28.19 ± 2.84 | -6.55 | 0.001* | 4.0 | 2.42 |

Per-seed accuracy (%), seeds 0–4: baseline 54.04 53.03 53.55 55.68 52.66; w1 50.98 52.04 52.04 53.50 48.87; nowl 49.25 52.24 50.58 52.19 46.40

## Does subgraph WL beat no WL at all?

Paired t-test of W1 against E1 "type on, WL off" (W1 minus WL off).

| Dataset | Δ acc | p (acc) | Δ F1 | p (F1) |
|---|---|---|---|---|
| R8 | -0.11 | 0.491 | -1.95 | 0.068 |
| Ohsumed | +1.36 | 0.036* | +3.39 | 0.002* |

## WL code statistics

Counted over all (slice, position) entries: 24 per document, all documents. "Unique" means the code occurs in exactly one entry. For full-graph WL a word keeps one code in every slice it appears in.

| Dataset | Scope | Entries | Distinct codes | Min code | Max code | Entries with a unique code |
|---|---|---|---|---|---|---|
| R8 | full | 184,176 | 15,205 | 0 | 15204 | 4.1% |
| R8 | subgraph | 184,176 | 156,156 | 0 | 156155 | 76.2% |
| Ohsumed | full | 177,600 | 21,511 | 0 | 21512 | 4.6% |
| Ohsumed | subgraph | 177,600 | 161,553 | 0 | 161552 | 84.3% |

Most common codes (code: count):

- **R8, full:** 15179: 3032, 15186: 2745, 15180: 2370, 15200: 2299, 15146: 2188, 15040: 2155, 14931: 1914, 15148: 1892, 15025: 1709, 15054: 1571
- **R8, subgraph:** 156135: 35, 149336: 20, 145279: 20, 156155: 19, 149820: 18, 150544: 16, 144679: 16, 144903: 16, 156153: 16, 154050: 16
- **Ohsumed, full:** 21368: 158, 21185: 141, 21348: 139, 21322: 126, 21079: 120, 21334: 120, 21105: 114, 21048: 112, 21008: 112, 20869: 111
- **Ohsumed, subgraph:** 161489: 14, 161551: 11, 161540: 11, 161530: 10, 161544: 10, 161053: 10, 161541: 9, 161536: 9, 87770: 9, 160332: 8

## Subgraph WL: refinement and cost

| Dataset | Rounds | Codes after each round | Edges per slice (min / mean / max) | Graph rebuild (s) | Subgraphs (s) | WL (s) | Total (s) |
|---|---|---|---|---|---|---|---|
| R8 | 5 | 1 → 23 → 151,806 → 156,123 → 156,156 → 156,156 | 44 / 188 / 272 | 1.6 | 1.0 | 4.9 | 7.5 |
| Ohsumed | 6 | 1 → 23 → 159,019 → 161,522 → 161,552 → 161,553 → 161,553 | 88 / 184 / 257 | 4.0 | 0.8 | 5.9 | 10.7 |

All slices are refined together with one dictionary per round, so codes are comparable across slices; the loop stops once no slice's partition changes. Each slice ends with the same partition as WL run on that slice alone (checked in the tests).

## Failures

None: all 2 datasets × 5 seeds completed.
