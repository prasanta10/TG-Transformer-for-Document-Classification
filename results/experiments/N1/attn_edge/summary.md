# TG-Transformer — experiments/N1/attn_edge

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False, attn_bias=edge; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.07 ± 0.34 | 86.38 ± 1.85 | 98.1 ± 0.1 | 5.6 | 15.6 | 3.56 |
| Ohsumed | 53.70 ± 1.20 | 34.70 ± 2.30 | 70.4 ± 0.4 | 5.6 | 15.6 | 2.57 |

Per-seed test accuracy (%):

- R8: 95.25 95.25 94.52 94.97 95.34
- Ohsumed: 54.04 52.86 53.18 55.65 52.78
