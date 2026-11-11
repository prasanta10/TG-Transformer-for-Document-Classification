# TG-Transformer — experiments/N1/attn_intimacy

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False, attn_bias=intimacy; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.05 ± 0.33 | 86.37 ± 1.86 | 98.1 ± 0.1 | 5.6 | 15.6 | 3.48 |
| Ohsumed | 53.57 ± 1.15 | 34.62 ± 2.35 | 70.4 ± 0.4 | 5.6 | 15.6 | 2.58 |

Per-seed test accuracy (%):

- R8: 95.25 95.11 94.52 94.97 95.39
- Ohsumed: 53.99 52.71 53.28 55.35 52.51
