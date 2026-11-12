"""E6 plots: test accuracy / macro-F1 vs depth, and loss curves for depths 2 and 5.

Reads results/experiments/e6/summary.csv (run summarize_e6.py first) and the seed-0
curves CSVs, and writes results/experiments/e6/e6_accuracy_vs_depth.png and
e6_curves_seed0.png.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, MaxNLocator  # noqa: E402

from plot_r_all import INK, INK2, MUTED, SURF, style  # noqa: E402
from run_experiments import DATASETS  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = "results/experiments/e6"
DEPTHS = [2, 3, 4, 5]
ACC, F1 = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated default palette)
D2, D5 = "#2a78d6", "#eb6834"


def accuracy_plot():
    """One panel per dataset: accuracy and macro-F1 (mean ± std over 5 seeds) vs depth."""
    with open(os.path.join(OUT, "summary.csv")) as f:
        rows = {(r["dataset"], int(r["depth"])): r for r in csv.DictReader(f)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURF)
    for ax, ds in zip(axes, DATASETS):
        style(ax)
        ax.xaxis.set_major_locator(FixedLocator(DEPTHS))
        for key, color, name, mk in (("acc", ACC, "Accuracy", "o"), ("f1", F1, "Macro-F1", "s")):
            m = [float(rows[(ds, L)][f"{key}_mean"]) for L in DEPTHS]
            sd = [float(rows[(ds, L)][f"{key}_std"]) for L in DEPTHS]
            ax.errorbar(DEPTHS, m, yerr=sd, color=color, linewidth=2, marker=mk, markersize=6,
                        capsize=4, elinewidth=1.2)
            ax.text(DEPTHS[-1] + 0.12, m[-1], name, color=INK2, fontsize=9, va="center")
        ax.set_xlim(1.7, 5.9)
        ax.set_xlabel("Number of Graph Transformer layers (2 = baseline)")
        ax.set_ylabel("Test score (%)", color=INK2)
        ax.set_title(f"{ds}: mean ± std over 5 seeds", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = os.path.join(OUT, "e6_accuracy_vs_depth.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    print("wrote", out)


def load(path):
    """Epochs, train loss and validation loss from a curves CSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return ([int(r["epoch"]) for r in rows], [float(r["train_loss"]) for r in rows],
            [float(r["val_loss"]) for r in rows])


def curves_plot():
    """One panel per dataset: train (solid) and validation (dashed) loss, depths 2 and 5, seed 0."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURF)
    for ax, ds in zip(axes, DATASETS):
        style(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for L, color in ((2, D2), (5, D5)):
            ep, tr, va = load(os.path.join(OUT, f"depth{L}", ds, "curves_seed_0.csv"))
            best = min(range(len(va)), key=va.__getitem__)
            ax.plot(ep, tr, color=color, linewidth=2)
            ax.plot(ep, va, color=color, linewidth=2, linestyle=(0, (4, 2)))
            ax.plot(ep[best], va[best], marker="o", color=color, markersize=7,
                    markeredgecolor=SURF, markeredgewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss", color=INK2)
        ax.set_title(f"{ds}, seed 0", loc="left", color=INK, fontsize=11)
    handles = [Line2D([], [], color=D2, linewidth=2, label="2 layers (baseline)"),
               Line2D([], [], color=D5, linewidth=2, label="5 layers"),
               Line2D([], [], color=MUTED, linewidth=2, label="train"),
               Line2D([], [], color=MUTED, linewidth=2, linestyle=(0, (4, 2)), label="validation"),
               Line2D([], [], color=MUTED, marker="o", linestyle="", label="best val loss")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = os.path.join(OUT, "e6_curves_seed0.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.chdir(ROOT)
    accuracy_plot()
    curves_plot()
