# TG-Transformer — experiments/e1/type_off_wl_off

Config: k=23, sampler=intimacy, type_enc=False, wl_enc=False, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 94.97 ± 0.42 | 83.82 ± 2.13 | 98.1 ± 0.1 | 3.6 | 13.6 | 3.19 |
| Ohsumed | 48.23 ± 1.60 | 26.52 ± 3.15 | 70.4 ± 0.4 | 4.0 | 14.0 | 2.32 |

Per-seed test accuracy (%):

- R8: 95.25 94.84 94.34 95.43 94.97
- Ohsumed: 48.21 48.95 48.85 45.51 49.64
