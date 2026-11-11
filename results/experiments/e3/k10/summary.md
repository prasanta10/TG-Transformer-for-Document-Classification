# TG-Transformer — experiments/e3/k10

Config: k=10, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 92.52 ± 0.50 | 83.40 ± 1.85 | 98.1 ± 0.1 | 5.6 | 15.6 | 2.61 |
| Ohsumed | 49.58 ± 1.02 | 34.12 ± 2.44 | 70.4 ± 0.4 | 6.0 | 16.0 | 2.05 |

Per-seed test accuracy (%):

- R8: 91.91 92.23 92.51 93.24 92.69
- Ohsumed: 48.53 49.27 50.90 50.38 48.83
