# TG-Transformer — experiments/e3/k40

Config: k=40, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.49 ± 0.22 | 87.13 ± 1.73 | 98.1 ± 0.1 | 6.8 | 16.8 | 4.69 |
| Ohsumed | 55.11 ± 1.58 | 33.66 ± 2.77 | 70.4 ± 0.4 | 5.4 | 15.4 | 3.19 |

Per-seed test accuracy (%):

- R8: 95.71 95.20 95.29 95.57 95.66
- Ohsumed: 55.06 54.84 55.21 57.46 53.01
