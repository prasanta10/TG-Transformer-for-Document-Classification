"""Summarise experiments E1-E5 against the reference configuration.

For every setting and dataset: test accuracy and macro-F1 (mean +/- sample std
over seeds 0-4), difference from the reference, and a two-sided paired t-test
across seeds (scipy.stats.ttest_rel; seeds pair up because each seed shares the
validation split and initialisation seed with the reference run).
Also draws the E3 plots and tabulates E2 sampler preprocessing times.

Outputs: results/experiments/summary.md, summary.csv, e3_accuracy_vs_k.png,
e3_time_vs_k.png
"""

import csv
import json
import os

import numpy as np
from scipy.stats import ttest_rel

from run_experiments import DATASETS, SEEDS, SETTINGS

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_CONFIG = "configs/baseline.yaml"      # configs/baseline_v2.yaml does not exist
REF_DIR = "results/baseline"
OUT = "results/experiments"
EXP_TITLES = {"e1": "E1: structural encodings", "e2": "E2: sampler", "e3": "E3: context size k",
              "e4": "E4: readout", "e5": "E5: embeddings"}
SETTING_LABELS = {
    "type_off_wl_off": "type off, WL off", "type_on_wl_off": "type on, WL off",
    "type_off_wl_on": "type off, WL on", "sampler_tfidf": "sampler = tfidf",
    "sampler_random": "sampler = random", "k5": "k = 5", "k10": "k = 10", "k15": "k = 15",
    "k30": "k = 30", "k40": "k = 40", "readout_target": "readout = target",
    "freeze_on": "freeze_embeddings = on",
}


def load_runs(base, ds):
    """Load seed_<s>.json for all seeds; return list (None where missing)."""
    runs = []
    for s in SEEDS:
        p = os.path.join(base, ds, f"seed_{s}.json")
        runs.append(json.load(open(p)) if os.path.exists(p) else None)
    return runs


def stats_row(runs, ref):
    """Mean/std of a setting, difference from the reference and paired t-test p-values."""
    acc = np.array([r["test_acc"] for r in runs]) * 100
    f1 = np.array([r["test_macro_f1"] for r in runs]) * 100
    racc = np.array([r["test_acc"] for r in ref]) * 100
    rf1 = np.array([r["test_macro_f1"] for r in ref]) * 100
    p_acc = ttest_rel(acc, racc).pvalue if not np.allclose(acc, racc) else 1.0
    p_f1 = ttest_rel(f1, rf1).pvalue if not np.allclose(f1, rf1) else 1.0
    return {"acc_mean": acc.mean(), "acc_std": acc.std(ddof=1),
            "f1_mean": f1.mean(), "f1_std": f1.std(ddof=1),
            "d_acc": acc.mean() - racc.mean(), "d_f1": f1.mean() - rf1.mean(),
            "p_acc": p_acc, "p_f1": p_f1,
            "epoch_time_s": float(np.mean([r["mean_epoch_time_s"] for r in runs])),
            "best_epoch": float(np.mean([r["best_epoch"] for r in runs])),
            "per_seed_acc": " ".join(f"{a:.2f}" for a in acc)}


def fmt_p(p):
    """p-value with a star when significant at 0.05."""
    return f"{p:.3f}*" if p < 0.05 else f"{p:.3f}"


def e3_plots(table):
    """Accuracy vs k (one panel per dataset) and time per epoch vs k (both datasets)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    COLORS = {"R8": "#2a78d6", "Ohsumed": "#eb6834"}   # categorical slots 1-2
    INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    ks = [5, 10, 15, 23, 30, 40]

    def series(ds, key):
        pts = []
        for k in ks:
            r = table.get(("reference", ds)) if k == 23 else table.get((f"k{k}", ds))
            if r is not None:
                pts.append((k, r[key], r.get(key.replace("mean", "std"), 0.0)))
        return pts

    def style(ax):
        ax.set_facecolor(SURF)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.set_major_locator(FixedLocator(ks))
        ax.set_xlabel("k (number of helper words; 23 = reference)", color=INK2)

    # accuracy: one panel per dataset (scales differ ~40 points), mean +/- std band
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
    for ax, ds in zip(axes, DATASETS):
        style(ax)
        pts = series(ds, "acc_mean")
        x, y, sd = (np.array(v) for v in zip(*pts))
        ax.fill_between(x, y - sd, y + sd, color=COLORS[ds], alpha=0.15, linewidth=0)
        ax.plot(x, y, color=COLORS[ds], linewidth=2, marker="o", markersize=5)
        ax.axvline(23, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(f"{ds}: test accuracy (%), mean ± std over 5 seeds", loc="left",
                     color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e3_accuracy_vs_k.png"), dpi=150, facecolor=SURF)
    plt.close(fig)

    # time per epoch: one axis, one line per dataset, direct labels + distinct markers
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor=SURF)
    style(ax)
    for ds, mk in zip(DATASETS, ("o", "s")):
        x, y, _ = zip(*series(ds, "epoch_time_s"))
        ax.plot(x, y, color=COLORS[ds], linewidth=2, marker=mk, markersize=6)
        ax.text(x[-1] + 0.8, y[-1], ds, color=INK2, fontsize=9, va="center")
    ax.set_xlim(3, 45)
    ax.set_ylim(bottom=0)
    ax.set_title("Mean training time per epoch (s, CPU)", loc="left", color=INK, fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e3_time_vs_k.png"), dpi=150, facecolor=SURF)
    plt.close(fig)


def main():
    """Build summary.md / summary.csv and plots."""
    os.chdir(ROOT)
    table, csv_rows, missing = {}, [], []
    ref = {ds: load_runs(REF_DIR, ds) for ds in DATASETS}
    for ds in DATASETS:
        r = ref[ds]
        acc = np.array([x["test_acc"] for x in r]) * 100
        f1 = np.array([x["test_macro_f1"] for x in r]) * 100
        table[("reference", ds)] = {
            "acc_mean": acc.mean(), "acc_std": acc.std(ddof=1), "f1_mean": f1.mean(),
            "f1_std": f1.std(ddof=1), "epoch_time_s": float(np.mean([x["mean_epoch_time_s"] for x in r])),
            "best_epoch": float(np.mean([x["best_epoch"] for x in r])),
            "per_seed_acc": " ".join(f"{a:.2f}" for a in acc)}
    for exp, setting, flags in SETTINGS:
        for ds in DATASETS:
            runs = load_runs(os.path.join(OUT, exp, setting), ds)
            if any(x is None for x in runs):
                missing.append(f"{exp}/{setting}/{ds} ({sum(x is None for x in runs)} seeds missing)")
                continue
            row = stats_row(runs, ref[ds])
            table[(setting, ds)] = row
            csv_rows.append({"experiment": exp, "setting": setting, "dataset": ds,
                             **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}})

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        fields = ["experiment", "setting", "dataset", "acc_mean", "acc_std", "d_acc", "p_acc",
                  "f1_mean", "f1_std", "d_f1", "p_f1", "best_epoch", "epoch_time_s", "per_seed_acc"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ds in DATASETS:
            rr = table[("reference", ds)]
            w.writerow({"experiment": "reference", "setting": "reference", "dataset": ds,
                        **{k: round(v, 4) if isinstance(v, float) else v for k, v in rr.items()}})
        for r in csv_rows:
            w.writerow({k: r.get(k) for k in fields})

    L = ["# Experiments E1–E5", "",
         f"**Reference configuration: `{REF_CONFIG}`** (`configs/baseline_v2.yaml` does not exist; "
         "the optional tuning step was not run). Reference results are reused from "
         f"`{REF_DIR}/`, not re-run: k = 23, intimacy sampler, type and WL encodings on, "
         "mean readout, fine-tuned embeddings.", "",
         "Every setting changes exactly one thing relative to the reference.",
         "",
         "- **Runs:** datasets R8 and Ohsumed; seeds 0–4.",
         "- **Validation split:** identical across settings for each seed (it depends only on the seed).",
         "- **Accuracy and macro-F1:** in %, mean ± sample std (ddof = 1) over the 5 seeds.",
         "- **Δ:** setting mean minus reference mean.",
         "- **p-values:** two-sided paired t-test across the 5 seeds (df = 4); `*` marks p < 0.05. No correction for multiple comparisons was applied (24 setting × dataset tests per metric).",
         ""]
    for ds in DATASETS:
        rr = table[("reference", ds)]
        L += [f"## {ds}", "",
              "| Experiment | Setting | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | s / epoch |",
              "|---|---|---|---|---|---|---|---|---|---|",
              f"| — | **reference** | {rr['acc_mean']:.2f} ± {rr['acc_std']:.2f} | — | — | "
              f"{rr['f1_mean']:.2f} ± {rr['f1_std']:.2f} | — | — | {rr['best_epoch']:.1f} | {rr['epoch_time_s']:.2f} |"]
        for exp, setting, _ in SETTINGS:
            r = table.get((setting, ds))
            if r is None:
                L.append(f"| {exp.upper()} | {SETTING_LABELS[setting]} | FAILED / missing | | | | | | | |")
                continue
            L.append(f"| {exp.upper()} | {SETTING_LABELS[setting]} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | "
                     f"{r['d_acc']:+.2f} | {fmt_p(r['p_acc'])} | {r['f1_mean']:.2f} ± {r['f1_std']:.2f} | "
                     f"{r['d_f1']:+.2f} | {fmt_p(r['p_f1'])} | {r['best_epoch']:.1f} | {r['epoch_time_s']:.2f} |")
        L.append("")

    # E3 plots
    if all((f"k{k}", ds) in table for k in (5, 10, 15, 30, 40) for ds in DATASETS):
        e3_plots(table)
        L += ["## E3: accuracy and time per epoch vs k", "",
              "![Accuracy vs k](e3_accuracy_vs_k.png)", "", "![Time per epoch vs k](e3_time_vs_k.png)", "",
              "The k = 23 point is the reference run from `results/baseline/`. The other k values ran one "
              "after another in a single batch on the same CPU, so their times per epoch are directly "
              "comparable. The k = 23 timing comes from the earlier baseline run, so it may differ "
              "slightly because of machine load.", ""]
    else:
        L += ["## E3 plots", "", "Not drawn: some E3 settings are missing.", ""]

    # E2 sampler timing
    tpath = os.path.join(OUT, "e2_sampler_timing.json")
    if os.path.exists(tpath):
        t = json.load(open(tpath))
        L += ["## E2: preprocessing time per sampler (k = 23, seconds)", "",
              "| Dataset | Sampler | Components | Total (s) |", "|---|---|---|---|"]
        for ds in DATASETS:
            for s in ("intimacy", "tfidf", "random"):
                comp = ", ".join(f"{k.removesuffix('_s')} {v:.3f}" for k, v in t[ds][s].items() if k != "total_s")
                L.append(f"| {ds} | {s} | {comp} | {t[ds][s]['total_s']:.2f} |")
        L += ["", "Intimacy times for the graph build, S and helper selection come from the preprocessing run "
              "(`results/preprocessing_<dataset>.json`). Everything else was measured by `time_samplers.py` "
              "(median of 3 repetitions). The WL encoding needs the graph whichever sampler is used, so the "
              "graph-build cost is avoided only if WL is also off.", ""]

    # failures
    log = json.load(open(os.path.join(OUT, "run_log.json"))) if os.path.exists(os.path.join(OUT, "run_log.json")) else {}
    failed = [f"{k} (exit {v['exit_code']}, see {v['log']})" for k, v in log.items() if not v["complete"]]
    L += ["## Failures", ""]
    L += [f"- {x}" for x in failed + missing] if (failed or missing) else ["None: all 12 settings × 2 datasets × 5 seeds completed."]
    L.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
