"""Entry point: preprocessing (cached) + training over datasets x seeds + summary.

Examples:
    python run.py --config configs/baseline.yaml                    # full baseline
    python run.py --config configs/baseline.yaml --preprocess_only
    python run.py --config configs/baseline.yaml --datasets R8 --seeds 0 \
        --max_epochs 1 --sampler tfidf --experiment smoke --no_save   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml

from attn_bias import get_attn_bias
from data import canonical_name, prepare_dataset
from wl_subgraph import get_wl_subgraph
from train import run_training

PAPER_ACC = {"R8": (98.1, 0.1), "Ohsumed": (70.4, 0.4)}


def str2bool(v: str) -> bool:
    """Parse on/off, true/false, yes/no, 1/0."""
    s = str(v).lower()
    if s in ("on", "true", "yes", "1"):
        return True
    if s in ("off", "false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected on/off, got '{v}'")


def parse_args(argv=None):
    """Command-line interface; every switch overrides the config file."""
    p = argparse.ArgumentParser(description="TG-Transformer baseline reimplementation")
    p.add_argument("--config", default="configs/baseline.yaml")
    p.add_argument("--experiment", help="results sub-directory name")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    # experiment switches
    p.add_argument("--k", type=int, help="E3: number of helper words (<= 50)")
    p.add_argument("--sampler", choices=["intimacy", "tfidf", "random"], help="E2")
    p.add_argument("--use_type_enc", type=str2bool, help="E1: on/off")
    p.add_argument("--use_wl_enc", type=str2bool, help="E1: on/off")
    p.add_argument("--readout", choices=["mean", "target"], help="E4")
    p.add_argument("--freeze_embeddings", type=str2bool, help="E5: on/off")
    p.add_argument("--attn_bias", choices=["none", "intimacy", "edge"],
                   help="N1: graph bias inside attention")
    p.add_argument("--wl_scope", choices=["full", "subgraph"],
                   help="W1: WL on the full graph or inside each slice")
    # training overrides
    p.add_argument("--max_epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--num_threads", type=int)
    # control
    p.add_argument("--preprocess_only", action="store_true")
    p.add_argument("--force_preprocess", action="store_true")
    p.add_argument("--no_save", action="store_true", help="do not write result files")
    p.add_argument("--log_train_acc", action="store_true",
                   help="also log eval-mode training accuracy each epoch (not timed)")
    return p.parse_args(argv)


def load_config(args) -> dict:
    """Load the YAML config and apply command-line overrides."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("experiment", "datasets", "seeds"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    for key in ("k", "sampler", "use_type_enc", "use_wl_enc", "readout", "freeze_embeddings",
                "attn_bias", "wl_scope"):
        if getattr(args, key) is not None:
            cfg["model"][key] = getattr(args, key)
    for key in ("max_epochs", "patience", "num_threads"):
        if getattr(args, key) is not None:
            cfg["train"][key] = getattr(args, key)
    if not 1 <= cfg["model"]["k"] <= cfg["preprocess"]["top_helpers"]:
        raise ValueError("k must be in [1, top_helpers]")
    cfg["datasets"] = [canonical_name(d) for d in cfg["datasets"]]
    return cfg


def write_summary(cfg: dict, all_results: dict, out_dir: str):
    """Write summary.md and summary.csv (mean +/- sample std over seeds, vs paper)."""
    rows = []
    for ds, res in all_results.items():
        acc = np.array([r["test_acc"] for r in res]) * 100
        f1 = np.array([r["test_macro_f1"] for r in res]) * 100
        ddof = 1 if len(res) > 1 else 0
        pa = PAPER_ACC.get(ds, (float("nan"), float("nan")))
        rows.append({
            "dataset": ds, "n_seeds": len(res),
            "acc_mean": acc.mean(), "acc_std": acc.std(ddof=ddof),
            "macro_f1_mean": f1.mean(), "macro_f1_std": f1.std(ddof=ddof),
            "paper_acc_mean": pa[0], "paper_acc_std": pa[1],
            "best_epoch_mean": np.mean([r["best_epoch"] for r in res]),
            "epochs_run_mean": np.mean([r["epochs_run"] for r in res]),
            "epoch_time_s_mean": np.mean([r["mean_epoch_time_s"] for r in res]),
            "per_seed_acc": " ".join(f"{a:.2f}" for a in acc),
        })
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    m = cfg["model"]
    lines = [
        f"# TG-Transformer — {cfg['experiment']}", "",
        f"Config: k={m['k']}, sampler={m['sampler']}, type_enc={m['use_type_enc']}, "
        f"wl_enc={m['use_wl_enc']}, readout={m['readout']}, "
        f"freeze_embeddings={m['freeze_embeddings']}, "
        f"attn_bias={m.get('attn_bias', 'none')}, wl_scope={m.get('wl_scope', 'full')}; seeds={cfg['seeds']}.", "",
        "Mean ± sample std (ddof=1) over seeds, in %.", "",
        "| Dataset | Test accuracy | Test macro-F1 | Paper accuracy | Best epoch (mean) "
        "| Epochs run (mean) | Time/epoch (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | "
            f"{r['macro_f1_mean']:.2f} ± {r['macro_f1_std']:.2f} | "
            f"{r['paper_acc_mean']:.1f} ± {r['paper_acc_std']:.1f} | "
            f"{r['best_epoch_mean']:.1f} | {r['epochs_run_mean']:.1f} | "
            f"{r['epoch_time_s_mean']:.2f} |")
    lines += ["", "Per-seed test accuracy (%):", ""]
    for r in rows:
        lines.append(f"- {r['dataset']}: {r['per_seed_acc']}")
    lines.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main(argv=None):
    """Run preprocessing and training as configured."""
    args = parse_args(argv)
    cfg = load_config(args)
    if cfg["train"].get("num_threads"):
        torch.set_num_threads(cfg["train"]["num_threads"])
    out_dir = os.path.join(cfg["paths"]["results_dir"], cfg["experiment"])
    all_results = {}
    for ds in cfg["datasets"]:
        data = prepare_dataset(ds, cfg, force=args.force_preprocess)
        if args.preprocess_only:
            continue
        ds_dir = os.path.join(out_dir, ds)
        if not args.no_save:
            os.makedirs(ds_dir, exist_ok=True)
        all_results[ds] = []
        for seed in cfg["seeds"]:
            print(f"\n=== {cfg['experiment']} | {ds} | seed {seed} ===", flush=True)
            bias = None
            if cfg["model"].get("attn_bias", "none") != "none":
                bias = get_attn_bias(data, cfg, seed)
            wl_sub = None
            if cfg["model"].get("wl_scope", "full") == "subgraph":
                wl_sub = get_wl_subgraph(data, cfg, seed)
            res, curves = run_training(data, cfg["model"], cfg["train"], seed, attn_bias=bias,
                                       wl_sub=wl_sub, log_train_acc=args.log_train_acc)
            if bias is not None:
                res["attn_bias_build"] = bias["info"]
            if wl_sub is not None:
                res["wl_subgraph_build"] = wl_sub["info"]
            res["config"] = {"model": cfg["model"], "train": cfg["train"],
                             "preprocess": cfg["preprocess"]}
            print(f"  -> test acc {100 * res['test_acc']:.2f}  macro-F1 "
                  f"{100 * res['test_macro_f1']:.2f}  best epoch {res['best_epoch']}/"
                  f"{res['epochs_run']}  {res['mean_epoch_time_s']:.2f}s/epoch", flush=True)
            all_results[ds].append(res)
            if args.no_save:
                continue
            with open(os.path.join(ds_dir, f"seed_{seed}.json"), "w") as f:
                json.dump(res, f, indent=2)
            with open(os.path.join(ds_dir, f"curves_seed_{seed}.csv"), "w", newline="") as f:
                fields = ["epoch", "train_loss", "val_loss", "val_acc", "epoch_time_s"]
                if args.log_train_acc:
                    fields += ["train_loss_plain", "train_acc"]
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(curves)
    if all_results and not args.no_save:
        write_summary(cfg, all_results, out_dir)


if __name__ == "__main__":
    sys.exit(main())
