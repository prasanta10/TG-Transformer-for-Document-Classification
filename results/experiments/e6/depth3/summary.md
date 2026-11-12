# TG-Transformer — experiments/e6/depth3

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False, attn_bias=none, wl_scope=full, num_layers=3; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 94.80 ± 0.32 | 86.34 ± 2.05 | 98.1 ± 0.1 | 6.8 | 16.8 | 3.90 |
| Ohsumed | 53.22 ± 1.04 | 34.77 ± 2.27 | 70.4 ± 0.4 | 5.2 | 15.2 | 2.90 |

Per-seed test accuracy (%):

- R8: 95.25 94.79 94.38 94.93 94.66
- Ohsumed: 54.98 53.20 52.86 52.26 52.78
