# TG-Transformer — experiments/e2/sampler_tfidf

Config: k=23, sampler=tfidf, type_enc=True, wl_enc=True, readout=mean, freeze_embeddings=False; seeds=[0, 1, 2, 3, 4].

Mean ± sample std (ddof=1) over seeds, in %.

| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) | Epochs run (mean) | Time/epoch (s) |
|---|---|---|---|---|---|---|
| R8 | 95.73 ± 0.31 | 88.28 ± 1.34 | 98.1 ± 0.1 | 10.8 | 20.8 | 3.28 |
| Ohsumed | 54.98 ± 1.02 | 35.48 ± 2.25 | 70.4 ± 0.4 | 5.6 | 15.6 | 2.38 |

Per-seed test accuracy (%):

- R8: 95.66 95.75 95.34 96.21 95.71
- Ohsumed: 55.65 54.49 54.71 56.32 53.72
