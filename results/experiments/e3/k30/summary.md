# TG-Transformer — experiments/e3/k30

Config: k=30, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.10 ± 0.55 | 86.55 ± 1.43 | 98.1 ± 0.1 | 11.0 | 21.0 | 3.89 |
| Ohsumed | 54.61 ± 1.73 | 34.74 ± 2.85 | 70.4 ± 0.4 | 5.6 | 15.6 | 2.76 |

Per-seed test accuracy (%):

- R8: 95.11 95.11 94.20 95.61 95.48
- Ohsumed: 55.85 54.12 53.80 56.81 52.46
