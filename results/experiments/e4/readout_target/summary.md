# TG-Transformer — experiments/e4/readout_target

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=target, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.69 ± 0.36 | 87.86 ± 2.01 | 98.1 ± 0.1 | 11.0 | 21.0 | 3.35 |
| Ohsumed | 45.47 ± 2.29 | 26.77 ± 2.75 | 70.4 ± 0.4 | 7.2 | 17.2 | 2.43 |

Per-seed test accuracy (%):

- R8: 95.29 96.03 96.07 95.39 95.66
- Ohsumed: 46.72 47.84 44.45 46.33 42.00
