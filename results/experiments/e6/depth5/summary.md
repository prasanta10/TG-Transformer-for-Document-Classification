# TG-Transformer — experiments/e6/depth5

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False, attn_bias=none, wl_scope=full, num_layers=5; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 94.78 ± 0.68 | 83.99 ± 2.52 | 98.1 ± 0.1 | 3.8 | 13.8 | 5.06 |
| Ohsumed | 53.67 ± 0.65 | 33.61 ± 1.82 | 70.4 ± 0.4 | 4.6 | 14.6 | 3.70 |

Per-seed test accuracy (%):

- R8: 95.48 95.48 94.11 94.70 94.15
- Ohsumed: 52.56 54.09 53.65 54.12 53.95
