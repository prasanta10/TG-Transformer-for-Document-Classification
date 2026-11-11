# TG-Transformer — experiments/e3/k15

Config: k=15, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 94.13 ± 0.43 | 85.62 ± 0.97 | 98.1 ± 0.1 | 6.4 | 16.4 | 2.81 |
| Ohsumed | 52.41 ± 0.93 | 35.59 ± 3.05 | 70.4 ± 0.4 | 5.8 | 15.8 | 2.18 |

Per-seed test accuracy (%):

- R8: 93.65 93.83 94.06 94.70 94.43
- Ohsumed: 51.35 52.29 53.28 53.43 51.69
