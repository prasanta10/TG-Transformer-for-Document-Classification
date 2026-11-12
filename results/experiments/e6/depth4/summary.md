# TG-Transformer — experiments/e6/depth4

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False, attn_bias=none, wl_scope=full, num_layers=4; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.24 ± 0.30 | 86.19 ± 3.11 | 98.1 ± 0.1 | 5.0 | 15.0 | 4.50 |
| Ohsumed | 53.36 ± 0.37 | 33.50 ± 1.13 | 70.4 ± 0.4 | 4.6 | 14.6 | 3.13 |

Per-seed test accuracy (%):

- R8: 95.71 95.20 95.20 94.88 95.20
- Ohsumed: 53.01 53.50 52.98 53.45 53.85
