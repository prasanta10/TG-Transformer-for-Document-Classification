# TG-Transformer — experiments/e1/type_off_wl_on

Config: k=23, sampler=intimacy, type_enc=False, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 94.99 ± 0.35 | 85.60 ± 3.11 | 98.1 ± 0.1 | 5.4 | 15.4 | 3.22 |
| Ohsumed | 53.12 ± 1.28 | 33.49 ± 3.80 | 70.4 ± 0.4 | 5.8 | 15.8 | 2.36 |

Per-seed test accuracy (%):

- R8: 95.29 94.93 94.43 95.07 95.25
- Ohsumed: 54.59 54.07 52.07 51.60 53.30
