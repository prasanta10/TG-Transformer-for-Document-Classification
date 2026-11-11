# TG-Transformer — experiments/W1/wl_subgraph

Config: k=23, sampler=intimacy, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False, attn_bias=none, wl_scope=subgraph; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 94.87 ± 0.32 | 84.21 ± 1.03 | 98.1 ± 0.1 | 5.2 | 15.2 | 3.38 |
| Ohsumed | 51.49 ± 1.71 | 31.58 ± 2.70 | 70.4 ± 0.4 | 5.6 | 15.6 | 2.49 |

Per-seed test accuracy (%):

- R8: 94.43 95.07 94.66 95.16 95.07
- Ohsumed: 50.98 52.04 52.04 53.50 48.87
