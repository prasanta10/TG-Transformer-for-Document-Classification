"""R-all: train vs validation loss for seed 0, baseline and R-all side by side.

Reads the curves CSVs of the baseline replay and of R-all (both logged with
--log_train_acc) and writes results/experiments/R_all/curves_<dataset>_seed0.png.
Both panels show plain cross-entropy (``train_loss_plain``), measured with dropout on
for the training set; R-all optimises a label-smoothed loss, whose values are not
comparable to the validation loss.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = "results/experiments/R_all"
TRAIN, VAL = "#2a78d6", "#eb6834"        # categorical slots 1-2 (validated default palette)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style(ax):
    """Recessive axes: hairline grid, muted ticks, no top/right spines."""
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.label.set_color(INK2)


def label_end(ax, x, y, text, color):
    """Direct label at the end of a line (text in ink, a short colored stub carries identity)."""
    ax.plot([x + 0.3, x + 0.9], [y, y], color=color, linewidth=2, solid_capstyle="round")
    ax.text(x + 1.1, y, text, color=INK2, fontsize=9, va="center")


def load(path):
    """Epochs, plain train loss and validation loss from a curves CSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return ([int(r["epoch"]) for r in rows], [float(r["train_loss_plain"]) for r in rows],
            [float(r["val_loss"]) for r in rows])


def plot(ds):
    """Two panels (baseline, R-all) of train vs validation loss for one dataset, seed 0."""
    runs = [("Baseline", os.path.join(OUT, "baseline_replay", ds, "curves_seed_0.csv")),
            ("R-all", os.path.join(OUT, ds, "curves_seed_0.csv"))]
    data = [(title, *load(p)) for title, p in runs]
    xmax = max(ep[-1] for _, ep, _, _ in data)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF, sharey=True)
    for ax, (title, ep, tr, va) in zip(axes, data):
        style(ax)
        best = min(range(len(va)), key=va.__getitem__) + 1
        ax.plot(ep, tr, color=TRAIN, linewidth=2, marker="o", markersize=4)
        ax.plot(ep, va, color=VAL, linewidth=2, linestyle=(0, (4, 2)), marker="o", markersize=4)
        ax.axvline(best, color=MUTED, linewidth=1, linestyle=":")
        ax.text(best, 1.02, f"best val loss (epoch {best})", transform=ax.get_xaxis_transform(),
                ha="center", color=MUTED, fontsize=8)
        ytr, yva = tr[-1], va[-1]
        gap = 0.06 * max(max(tr), max(va))      # keep the two end labels apart
        if abs(ytr - yva) < gap:
            mid, up = (ytr + yva) / 2, 1 if ytr >= yva else -1
            ytr, yva = mid + up * gap / 2, mid - up * gap / 2
        label_end(ax, ep[-1], ytr, "Train", TRAIN)
        label_end(ax, ep[-1], yva, "Validation", VAL)
        ax.set_xlim(0.5, xmax + 6.5)
        ax.set_xlabel("Epoch")
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=16)
    axes[0].set_ylabel("Cross-entropy loss", color=INK2)
    fig.suptitle(f"{ds}, seed 0: train vs validation loss (validation line dashed)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUT, f"curves_{ds}_seed0.png")
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    os.chdir(ROOT)
    for ds in sys.argv[1:] or ["R8", "Ohsumed"]:
        plot(ds)
