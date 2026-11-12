"""Summarise E6 (depth: number of Graph Transformer layers 2-5) against the baseline.

Baseline (depth 2) accuracy and macro-F1 come from results/baseline/, the same
reference as E1-E5. Depth 2 was re-run in results/experiments/e6/depth2/ with
training-accuracy and over-smoothing logging; it is checked to reproduce the saved
baseline exactly (test accuracy and validation-loss curves), and supplies the
depth-2 train accuracy, parameter counts and over-smoothing values.

Outputs: results/experiments/e6/summary.md, summary.csv (one row per depth x
dataset), per_seed.csv (one row per depth x dataset x seed).
"""

import csv
import json
import os

import numpy as np

from run_experiments import DATASETS, SEEDS
from summarize_experiments import fmt_p, load_runs, stats_row

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_DIR = "results/baseline"
OUT = "results/experiments/e6"
DEPTHS = [2, 3, 4, 5]
OS_SEED = 0                         # over-smoothing is reported for this seed

# Written after inspecting the results (see the tables below for the numbers).
FINDINGS = [
    "- **Depth does nothing measurable.** Across 3, 4 and 5 layers every accuracy change is "
    "within ±0.6 points and no difference is significant (all p ≥ 0.07, accuracy and "
    "macro-F1, both datasets). The largest drop is R8 macro-F1 at 5 layers (−2.5, p = 0.19), "
    "driven by two seeds; it comes with a doubled seed spread (accuracy std 0.29 → 0.68).",
    "- **Fit and overfitting are unchanged.** Training accuracy at the best epoch stays at "
    "99.3–99.6% on R8 and 87.6–89.8% on Ohsumed. Deeper models reach their best validation "
    "loss as early or earlier (R8 best epoch 5.6 → 3.8 at 5 layers; Ohsumed 5.6 → 4.6) and "
    "fit the training set slightly faster (seed-0 curves), so the extra capacity only "
    "speeds up overfitting.",
    "- **Mild, gradual smoothing, no collapse.** Mean pairwise cosine similarity of the "
    "output node vectors rises with depth (R8 0.66 → 0.73, Ohsumed 0.45 → 0.58 from 2 to "
    "5 layers) and, inside the 5-layer models, rises layer by layer (per_seed.csv). It stays "
    "far from 1; before the G-Res term it is ~0.12 higher, so G-Res keeps the nodes apart. "
    "The smoothing does not show up as an accuracy loss.",
    "- **Cost:** each extra layer adds 6,464 parameters (+20% of the 33k non-embedding "
    "parameters; negligible next to the 4.6–6.5 M embedding table) and ~0.5 s / epoch; "
    "5 layers is ~1.5× the baseline time per epoch (R8 3.41 → 5.06 s, Ohsumed 2.48 → 3.70 s) "
    "for no gain. Two layers remains the right choice.",
]


def val_losses(base, ds, s):
    """Validation-loss curve of one run."""
    with open(os.path.join(base, ds, f"curves_seed_{s}.csv")) as f:
        return [float(r["val_loss"]) for r in csv.DictReader(f)]


def cell(r, key, base=False):
    """'mean ± std' for the baseline, 'mean ± std (Δ)' with * if p < 0.05 otherwise."""
    m, sd = r[f"{key}_mean"], r[f"{key}_std"]
    if base:
        return f"{m:.2f} ± {sd:.2f}"
    star = "*" if r[f"p_{key}"] < 0.05 else ""
    return f"{m:.2f} ± {sd:.2f} ({r[f'd_{key}']:+.2f}){star}"


def main():
    """Build the E6 tables and CSV files."""
    os.chdir(ROOT)
    rows, missing, replay_ok, per_seed = {}, [], {}, []
    for ds in DATASETS:
        ref = load_runs(REF_DIR, ds)
        for L in DEPTHS:
            base = os.path.join(OUT, f"depth{L}")
            runs = load_runs(base, ds)
            if any(r is None for r in runs):
                missing.append(f"e6/depth{L}/{ds} ({sum(r is None for r in runs)} seeds missing)")
                continue
            if L == 2:
                replay_ok[ds] = all(a["test_acc"] == b["test_acc"] and
                                    val_losses(REF_DIR, ds, s) == val_losses(base, ds, s)
                                    for s, (a, b) in enumerate(zip(ref, runs)))
            # depth 2: accuracy / F1 / time / best epoch from the saved baseline
            st = stats_row(ref if L == 2 else runs, ref)
            tr = np.array([r["train_acc_at_best"] for r in runs]) * 100
            os_ = runs[OS_SEED]["oversmoothing_test"]
            rows[(ds, L)] = {
                **st,
                "train_acc_mean": tr.mean(), "train_acc_std": tr.std(ddof=1),
                "epochs_run": float(np.mean([r["epochs_run"] for r in runs])),
                "params_total": runs[0]["n_trainable_params"],
                "params_no_emb": runs[0]["n_trainable_params_no_emb"],
                "epoch_time_s": float(np.mean([r["mean_epoch_time_s"] for r in runs])),
                "cos_input": os_["input"], "cos_last": os_[f"layer_{L}"],
                "cos_last_pre_res": os_["last_pre_res"],
                "cos_last_all_seeds": float(np.mean([r["oversmoothing_test"][f"layer_{L}"]
                                                     for r in runs])),
            }
            for r in runs:
                o = r["oversmoothing_test"]
                per_seed.append({
                    "dataset": ds, "depth": L, "seed": r["seed"], "test_acc": r["test_acc"],
                    "test_macro_f1": r["test_macro_f1"], "best_epoch": r["best_epoch"],
                    "epochs_run": r["epochs_run"], "train_acc_at_best": r["train_acc_at_best"],
                    "val_acc_at_best": r["val_acc_at_best"],
                    "mean_epoch_time_s": r["mean_epoch_time_s"],
                    "total_train_time_s": r["total_train_time_s"],
                    "n_trainable_params": r["n_trainable_params"],
                    "n_trainable_params_no_emb": r["n_trainable_params_no_emb"],
                    "cos_input": o["input"], "cos_last": o[f"layer_{L}"],
                    "cos_last_pre_res": o["last_pre_res"],
                    "cos_per_layer": " ".join(f"{o[f'layer_{i}']:.4f}" for i in range(1, L + 1)),
                })

    with open(os.path.join(OUT, "per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0]))
        w.writeheader()
        w.writerows(per_seed)
    fields = ["dataset", "depth", "acc_mean", "acc_std", "d_acc", "p_acc", "f1_mean", "f1_std",
              "d_f1", "p_f1", "best_epoch", "epochs_run", "train_acc_mean", "train_acc_std",
              "epoch_time_s", "params_total", "params_no_emb", "cos_input", "cos_last",
              "cos_last_pre_res", "cos_last_all_seeds", "per_seed_acc"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (ds, L), r in rows.items():
            w.writerow({"dataset": ds, "depth": L,
                        **{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in r.items() if k in fields}})

    L_ = ["# E6 · Depth: number of Graph Transformer layers", "",
          "One change relative to `configs/baseline.yaml`: `num_layers` ∈ {3, 4, 5} "
          "(baseline 2), via `run.py --num_layers`. Everything else is the baseline: intimacy "
          "sampler (α = 0.15, k = 23, cached slices), type + full-graph WL encodings, hidden 32, "
          "4 heads, mean readout, Adam lr 1e-3, weight decay 5e-4 (not on the embedding table), "
          "dropout 0.5, batch 64, ≤ 200 epochs, early stopping patience 10 on validation loss, "
          "fine-tuned embeddings. Seeds 0–4 with the same validation splits as the baseline. "
          "Each layer has its own attention / FFN weights; the G-Res projection of the raw "
          "features is shared and added after every layer; readout and classifier are "
          "unchanged.", "",
          f"Baseline accuracy and macro-F1 come from `{REF_DIR}/`. Depth 2 was re-run with "
          f"the E6 code (`{OUT}/depth2/`) for train accuracy and over-smoothing; it "
          "reproduced the saved baseline exactly: "
          + ", ".join(f"{ds} {'yes' if ok else 'NO'}" for ds, ok in replay_ok.items()) + ".", "",
          "- **Accuracy and macro-F1:** test set, in %, mean ± sample std (ddof = 1) over "
          "seeds 0–4.",
          "- **(Δ):** depth mean minus baseline mean; `*` marks p < 0.05 in a two-sided paired "
          "t-test across the 5 seeds (df = 4). No multiple-comparison correction.", "",
          "## Results", "",
          "| Setting | R8 accuracy | R8 macro-F1 | Ohsumed accuracy | Ohsumed macro-F1 |",
          "|---|---|---|---|---|"]
    for L in DEPTHS:
        label = "**baseline (2 layers)**" if L == 2 else f"{L} layers"
        cells = []
        for ds in DATASETS:
            r = rows.get((ds, L))
            cells += ["missing", "missing"] if r is None else \
                [cell(r, "acc", L == 2), cell(r, "f1", L == 2)]
        L_.append(f"| {label} | " + " | ".join(cells) + " |")
    L_ += ["", "p-values (accuracy / macro-F1): " + "; ".join(
        f"{ds} {L} layers {rows[(ds, L)]['p_acc']:.3f} / {rows[(ds, L)]['p_f1']:.3f}"
        for ds in DATASETS for L in DEPTHS[1:] if (ds, L) in rows), "",
        "![Test accuracy and macro-F1 vs depth](e6_accuracy_vs_depth.png)", "",
        "## Cost, fit and over-smoothing", "",
        "- **Params:** trainable parameters, total / without the node-embedding table "
        "(the table is the same at every depth).",
        "- **s / epoch:** mean wall time of one epoch (training + validation), mean over seeds; "
        "CPU, runs executed one after another.",
        "- **Train acc at best:** eval mode (dropout off), training documents without the "
        "validation split, at the best-validation-loss epoch; mean ± std over seeds.",
        f"- **Cos sim:** mean pairwise cosine similarity between the real nodes of a test "
        f"slice (24 nodes, 276 pairs), averaged over all test slices, seed {OS_SEED}, best "
        "checkpoint. *Output* = after the last layer including G-Res (what the readout "
        "averages); *pre-G-Res* = last encoder layer before the G-Res term is added; *input* = "
        "projected H(0) before layer 1. 1.0 would mean all node vectors point the same way.", ""]
    for ds in DATASETS:
        L_ += [f"### {ds}", "",
               "| Depth | Params (total / no emb) | s / epoch | Best epoch | Epochs run | "
               "Train acc at best | Cos sim output | Cos sim pre-G-Res | Cos sim input |",
               "|---|---|---|---|---|---|---|---|---|"]
        for L in DEPTHS:
            r = rows.get((ds, L))
            if r is None:
                L_.append(f"| {L} | missing | | | | | | | |")
                continue
            L_.append(f"| {L} | {r['params_total']:,} / {r['params_no_emb']:,} | "
                      f"{r['epoch_time_s']:.2f} | {r['best_epoch']:.1f} | {r['epochs_run']:.1f} | "
                      f"{r['train_acc_mean']:.2f} ± {r['train_acc_std']:.2f} | "
                      f"{r['cos_last']:.3f} | {r['cos_last_pre_res']:.3f} | {r['cos_input']:.3f} |")
        L_.append("")
    L_ += ["Per-seed test accuracy (%), seeds 0–4: " + "; ".join(
        f"{ds} {L} layers: {rows[(ds, L)]['per_seed_acc']}"
        for ds in DATASETS for L in DEPTHS if (ds, L) in rows), "",
        "![Train vs validation loss, depths 2 and 5, seed 0](e6_curves_seed0.png)", "",
        "Training loss is measured with dropout on and averaged over the epoch; validation "
        "loss with dropout off.", ""]
    if FINDINGS:
        L_ += ["## Summary", ""] + FINDINGS + [""]

    log_path = "results/experiments/run_log.json"
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    failed = [f"{k} (exit {v['exit_code']}, see {v['log']})" for k, v in log.items()
              if k.startswith("e6/") and not v["complete"]]
    L_ += ["## Failures", ""]
    L_ += [f"- {x}" for x in failed + missing] if (failed or missing) else \
        [f"None: all {len(DEPTHS)} depths completed for {len(DATASETS)} datasets × "
         f"{len(SEEDS)} seeds."]
    L_.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L_))
    print("\n".join(L_))


if __name__ == "__main__":
    main()
