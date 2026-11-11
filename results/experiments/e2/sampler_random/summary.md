# TG-Transformer — experiments/e2/sampler_random

Config: k=23, sampler=random, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 85.59 ± 1.41 | 51.41 ± 5.43 | 98.1 ± 0.1 | 8.6 | 18.6 | 3.22 |
| Ohsumed | 14.66 ± 0.18 | 1.29 ± 0.39 | 70.4 ± 0.4 | 2.4 | 12.4 | 2.54 |

Per-seed test accuracy (%):

- R8: 86.80 85.11 85.56 83.51 86.98
- Ohsumed: 14.59 14.99 14.59 14.57 14.57
