# TG-Transformer — experiments/e5/freeze_on

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=True; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.05 ± 0.32 | 87.73 ± 1.25 | 98.1 ± 0.1 | 8.8 | 18.8 | 1.50 |
| Ohsumed | 58.36 ± 1.25 | 42.17 ± 3.02 | 70.4 ± 0.4 | 10.8 | 20.8 | 0.93 |

Per-seed test accuracy (%):

- R8: 95.07 94.97 94.70 94.93 95.57
- Ohsumed: 56.81 59.73 59.53 58.05 57.66
