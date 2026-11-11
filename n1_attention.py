"""N1 (optional analysis): target-to-helper attention by intimacy rank, seed 0, test set.

Retrains seed 0 of the reference (attn_bias = none) and of N1a (attn_bias = intimacy)
on each dataset. Training is deterministic, so the retrained models must reproduce
the stored test accuracy; this is checked. With the best checkpoint loaded and
dropout off, the attention weights from the target position (query 0) to the helper
positions are averaged over test documents, heads and layers, and grouped by the
helper's intimacy rank (helpers are stored in descending intimacy order).

Output: results/experiments/N1/attention_by_rank.json
"""

import copy
import json
import os

import numpy as np
import torch
import yaml

from attn_bias import get_attn_bias
from data import load_prepared
from sampling import build_slices
from train import run_training

ROOT = os.path.dirname(os.path.abspath(__file__))
GROUPS = {"1-5": (1, 5), "6-12": (6, 12), "13-23": (13, 23)}
SETTINGS = {"reference": ("none", "results/baseline"),
            "attn_intimacy": ("intimacy", "results/experiments/N1/attn_intimacy")}


@torch.no_grad()
def attention_by_rank(model, data, cfg, bias, idx, bs=256):
    """Mean attention weight from the target to helpers of each rank group, plus to itself."""
    ids, mask, _ = build_slices(data, cfg["model"]["sampler"], cfg["model"]["k"], 0)
    ids, mask = torch.as_tensor(ids), torch.as_tensor(mask)
    B = None if bias is None else torch.as_tensor(bias["B"])
    model.eval()
    for layer in model.layers:
        layer.attn.keep_attn = True
    tot = torch.zeros(ids.shape[1], dtype=torch.float64)
    for s in range(0, len(idx), bs):
        b = torch.as_tensor(idx[s:s + bs])
        model(ids[b], mask[b], None if B is None else B[b])
        for layer in model.layers:               # (batch, heads, L, L); query 0 = target
            tot += layer.attn.last_attn[:, :, 0, :].sum(dim=(0, 1)).double()
    n = len(idx) * model.layers[0].attn.h * len(model.layers)
    per_pos = (tot / n).numpy()                  # mean over docs, heads, layers
    out = {"self": float(per_pos[0])}
    for g, (lo, hi) in GROUPS.items():
        out[f"mean_per_helper_{g}"] = float(per_pos[lo:hi + 1].mean())
        out[f"total_{g}"] = float(per_pos[lo:hi + 1].sum())
    return out


def main():
    """Retrain seed 0 for the reference and N1a on both datasets and record attention."""
    os.chdir(ROOT)
    cfg0 = yaml.safe_load(open("configs/baseline.yaml"))
    res = {}
    for ds in ("R8", "Ohsumed"):
        data = load_prepared(cfg0["paths"]["processed_dir"], ds)
        test_idx = np.where(data.is_test)[0]
        for name, (mode, rdir) in SETTINGS.items():
            cfg = copy.deepcopy(cfg0)
            cfg["model"]["attn_bias"] = mode
            bias = get_attn_bias(data, cfg, 0) if mode != "none" else None
            print(f"=== {ds} | {name} | seed 0 ===", flush=True)
            r, _, model = run_training(data, cfg["model"], cfg["train"], 0, log_every=0,
                                       attn_bias=bias, return_model=True)
            stored = json.load(open(os.path.join(rdir, ds, "seed_0.json")))["test_acc"]
            row = attention_by_rank(model, data, cfg, bias, test_idx)
            row.update({"test_acc_retrained": r["test_acc"], "test_acc_stored": stored,
                        "reproduced": bool(abs(r["test_acc"] - stored) < 1e-12)})
            res.setdefault(ds, {})[name] = row
            print(json.dumps(row, indent=1), flush=True)
    out = "results/experiments/N1/attention_by_rank.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
