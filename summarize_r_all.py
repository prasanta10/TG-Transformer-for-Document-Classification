"""Summarise R-all (four anti-overfitting remedies at once) against the baseline.

Baseline numbers come from results/baseline/. Training accuracy at the best epoch is
not stored there, so it comes from a replay of the baseline with training-accuracy
logging (results/experiments/R_all/baseline_replay/); the replay is checked to be
identical to the saved baseline (test accuracy and validation-loss curves).

Outputs: results/experiments/R_all/summary.md, summary.csv
"""

import csv
import json
import os

import numpy as np

from run_experiments import DATASETS, SEEDS
from summarize_experiments import fmt_p, load_runs, stats_row

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_DIR = "results/baseline"
OUT = "results/experiments/R_all"
REPLAY_DIR = os.path.join(OUT, "baseline_replay")
PAPER_ACC = {"R8": 98.1, "Ohsumed": 70.4}


def val_losses(base, ds, s):
    """Validation-loss curve of one run."""
    with open(os.path.join(base, ds, f"curves_seed_{s}.csv")) as f:
        return [float(r["val_loss"]) for r in csv.DictReader(f)]


def main():
    """Build summary.md / summary.csv for R-all."""
    os.chdir(ROOT)
    rows, missing, replay_ok = {}, [], {}
    for ds in DATASETS:
        ref, rep, rall = load_runs(REF_DIR, ds), load_runs(REPLAY_DIR, ds), load_runs(OUT, ds)
        if any(r is None for r in rep):
            missing.append(f"R_all/baseline_replay/{ds}")
        else:
            replay_ok[ds] = all(a["test_acc"] == b["test_acc"] and
                                val_losses(REF_DIR, ds, s) == val_losses(REPLAY_DIR, ds, s)
                                for s, (a, b) in enumerate(zip(ref, rep)))
            rows[(ds, "baseline")] = {**stats_row(ref, ref),
                                      "train_acc": np.mean([r["train_acc_at_best"] for r in rep]) * 100,
                                      "train_acc_std": np.std([r["train_acc_at_best"] for r in rep], ddof=1) * 100,
                                      "val_acc": np.mean([r["val_acc_at_best"] for r in rep]) * 100,
                                      "val_acc_std": np.std([r["val_acc_at_best"] for r in rep], ddof=1) * 100}
        if any(r is None for r in rall):
            missing.append(f"R_all/{ds} ({sum(r is None for r in rall)} seeds missing)")
            continue
        rows[(ds, "r_all")] = {**stats_row(rall, ref),
                               "train_acc": np.mean([r["train_acc_at_best"] for r in rall]) * 100,
                               "train_acc_std": np.std([r["train_acc_at_best"] for r in rall], ddof=1) * 100,
                               "val_acc": np.mean([r["val_acc_at_best"] for r in rall]) * 100,
                               "val_acc_std": np.std([r["val_acc_at_best"] for r in rall], ddof=1) * 100,
                               "epochs_run": float(np.mean([r["epochs_run"] for r in rall]))}
        rows[(ds, "baseline")]["epochs_run"] = float(np.mean([r["epochs_run"] for r in ref]))

    fields = ["dataset", "setting", "acc_mean", "acc_std", "d_acc", "p_acc", "f1_mean", "f1_std",
              "d_f1", "p_f1", "best_epoch", "epochs_run", "train_acc", "train_acc_std", "val_acc",
              "val_acc_std", "epoch_time_s", "per_seed_acc"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ds in DATASETS:
            for key in ("baseline", "r_all"):
                if (ds, key) in rows:
                    w.writerow({"dataset": ds, "setting": key,
                                **{k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in rows[(ds, key)].items() if k in fields}})
            w.writerow({"dataset": ds, "setting": "paper", "acc_mean": PAPER_ACC[ds]})

    L = ["# R-all: four anti-overfitting remedies at once", "",
         "R-all is `configs/baseline.yaml` with exactly four changes, fixed in advance and not "
         "tuned: weight decay 5e-3 (baseline 5e-4), label smoothing 0.1 on the training loss "
         "(baseline 0), training helpers redrawn every epoch from each document's top-50 "
         "intimacy list with probability proportional to intimacy (baseline: fixed top-23), "
         "and frozen input embeddings (baseline: fine-tuned). Validation and test always use "
         "the fixed top-23 helpers, and validation loss for early stopping is plain "
         "cross-entropy. Same seeds, validation splits and early stopping as the baseline.", "",
         f"Baseline accuracy, macro-F1, best epoch and s / epoch come from `{REF_DIR}/`. "
         "Train and validation accuracy at the best epoch come from a replay of the baseline "
         f"with training-accuracy logging (`{REPLAY_DIR}/`), which reproduced the saved "
         "baseline exactly: "
         + ", ".join(f"{ds} {'yes' if ok else 'NO'}" for ds, ok in replay_ok.items()) + ".", "",
         "- **Accuracy and macro-F1:** test set, in %, mean ± sample std (ddof = 1) over seeds 0–4.",
         "- **Train accuracy:** eval mode (dropout off), training documents without the "
         "validation split, fixed top-23 helpers, at the best-validation-loss epoch.",
         "- **Δ:** R-all mean minus baseline mean. **p:** two-sided paired t-test across the "
         "5 seeds (df = 4); `*` marks p < 0.05.",
         "- **Paper:** TG-Transformer accuracy reported in the paper; macro-F1 is not reported.", ""]
    for ds in DATASETS:
        L += [f"## {ds}", "",
              "| Model | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | "
              "Epochs run | Train acc at best | Val acc at best | s / epoch |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, label in (("baseline", "**baseline**"), ("r_all", "R-all")):
            r = rows.get((ds, key))
            if r is None:
                L.append(f"| {label} | FAILED / missing | | | | | | | | | | |")
                continue
            d = ("— | — |", "— | — |") if key == "baseline" else \
                (f"{r['d_acc']:+.2f} | {fmt_p(r['p_acc'])} |", f"{r['d_f1']:+.2f} | {fmt_p(r['p_f1'])} |")
            L.append(f"| {label} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | {d[0]} "
                     f"{r['f1_mean']:.2f} ± {r['f1_std']:.2f} | {d[1]} {r['best_epoch']:.1f} | "
                     f"{r['epochs_run']:.1f} | {r['train_acc']:.2f} ± {r['train_acc_std']:.2f} | "
                     f"{r['val_acc']:.2f} ± {r['val_acc_std']:.2f} | {r['epoch_time_s']:.2f} |")
        L.append(f"| paper (TG-Transformer) | {PAPER_ACC[ds]:.1f} | | | not reported | | | | | | | |")
        L += ["", "Per-seed test accuracy (%), seeds 0–4: " + "; ".join(
            f"{key} {rows[(ds, key)]['per_seed_acc']}" for key in ("baseline", "r_all")
            if (ds, key) in rows), "",
            f"![Train vs validation loss, seed 0](curves_{ds}_seed0.png)", ""]
    L += ["The loss curves show plain cross-entropy for both models. R-all optimises the "
          "label-smoothed loss, which cannot fall below the entropy of the smoothed target "
          "(0.47 for R8's 8 classes, 0.61 for Ohsumed's 23), so its plain training loss is "
          "plotted to keep the panels comparable. Training loss is measured with dropout on, "
          "averaged over the epoch.", ""]

    log_path = "results/experiments/run_log.json"
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    failed = [f"{k} (exit {v['exit_code']}, see {v['log']})" for k, v in log.items()
              if k.startswith("R_all/") and not v["complete"]]
    L += ["## Failures", ""]
    L += [f"- {x}" for x in failed + missing] if (failed or missing) else \
        [f"None: R-all and the baseline replay completed for all {len(DATASETS)} datasets × "
         f"{len(SEEDS)} seeds."]
    L.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
