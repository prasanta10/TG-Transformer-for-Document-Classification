# TG-Transformer — experiments/e6/depth2

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False, attn_bias=none, wl_scope=full, num_layers=2; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.08 ± 0.29 | 86.50 ± 1.76 | 98.1 ± 0.1 | 5.6 | 15.6 | 3.41 |
| Ohsumed | 53.79 ± 1.18 | 34.74 ± 2.43 | 70.4 ± 0.4 | 5.6 | 15.6 | 2.48 |

Per-seed test accuracy (%):

- R8: 95.29 95.11 94.61 95.02 95.34
- Ohsumed: 54.04 53.03 53.55 55.68 52.66
