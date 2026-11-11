# TG-Transformer — experiments/e1/type_on_wl_off

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=False, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 94.98 ± 0.41 | 86.15 ± 0.89 | 98.1 ± 0.1 | 5.4 | 15.4 | 3.32 |
| Ohsumed | 50.13 ± 2.43 | 28.19 ± 2.84 | 70.4 ± 0.4 | 4.0 | 14.0 | 2.42 |

Per-seed test accuracy (%):

- R8: 94.97 95.20 94.29 95.29 95.16
- Ohsumed: 49.25 52.24 50.58 52.19 46.40
