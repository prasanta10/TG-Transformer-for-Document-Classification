# TG-Transformer — experiments/e3/k5

Config: k=5, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 89.78 ± 0.53 | 78.42 ± 1.27 | 98.1 ± 0.1 | 4.2 | 14.2 | 2.52 |
| Ohsumed | 43.44 ± 1.29 | 28.36 ± 3.83 | 70.4 ± 0.4 | 5.2 | 15.2 | 1.74 |

Per-seed test accuracy (%):

- R8: 90.09 88.94 89.63 90.32 89.90
- Ohsumed: 43.58 41.33 43.71 44.89 43.68
