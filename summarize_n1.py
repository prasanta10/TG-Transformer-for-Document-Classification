"""Summarise novel experiment N1 (edge-aware attention) against the reference.

Same statistics as ``summarize_experiments.py`` (mean +/- sample std over seeds 0-4,
difference from the reference, two-sided paired t-test across seeds), plus the
learned beta values, the cost of building B and, if present, the attention-by-rank
analysis from ``n1_attention.py``.

Outputs: results/experiments/N1/summary.md, summary.csv
"""

import csv
import json
import os

import numpy as np

from run_experiments import DATASETS, N1_SETTINGS, SEEDS
from summarize_experiments import fmt_p, load_runs, stats_row

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_DIR = "results/baseline"
OUT = "results/experiments/N1"
LABELS = {"attn_intimacy": "N1a: attn_bias = intimacy", "attn_edge": "N1b: attn_bias = edge"}


def main():
    """Build summary.md / summary.csv for N1."""
    os.chdir(ROOT)
    ref = {ds: load_runs(REF_DIR, ds) for ds in DATASETS}
    rows, runs_by, missing = {}, {}, []
    for _, setting, _ in N1_SETTINGS:
        for ds in DATASETS:
            runs = load_runs(os.path.join(OUT, setting), ds)
            if any(r is None for r in runs):
                missing.append(f"N1/{setting}/{ds} ({sum(r is None for r in runs)} seeds missing)")
                continue
            rows[(setting, ds)] = stats_row(runs, ref[ds])
            runs_by[(setting, ds)] = runs

    fields = ["setting", "dataset", "acc_mean", "acc_std", "d_acc", "p_acc", "f1_mean", "f1_std",
              "d_f1", "p_f1", "best_epoch", "epoch_time_s", "per_seed_acc"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ds in DATASETS:
            rr = stats_row(ref[ds], ref[ds])
            w.writerow({"setting": "reference", "dataset": ds,
                        **{k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in rr.items() if k in fields}})
            for _, setting, _ in N1_SETTINGS:
                if (setting, ds) in rows:
                    w.writerow({"setting": setting, "dataset": ds,
                                **{k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in rows[(setting, ds)].items() if k in fields}})

    L = ["# N1: edge-aware attention", "",
         "Attention scores get a graph bias: softmax(QKᵀ/√d_h + β[l,h]·B + mask). B is a 24 × 24 "
         "z-scored matrix of graph scores for each document's slice, and β is one learnable "
         "scalar per layer and head (2 × 4), initialised to 0 and excluded from weight decay.", "",
         "- **N1a (`intimacy`):** B from log(S + 1e-10), S = intimacy (paper Eq. 2).",
         "- **N1b (`edge`):** B from log(1 + edge weight), TF-IDF for target–helper pairs and "
         "positive PMI for helper–helper pairs, 0 without an edge.", "",
         f"Reference: `configs/baseline.yaml`, results reused from `{REF_DIR}/`. Everything except "
         "the attention bias is identical (k = 23, intimacy sampler, type and WL encodings on, "
         "mean readout, fine-tuned embeddings, same seeds and validation splits).", "",
         "- **Accuracy and macro-F1:** in %, mean ± sample std (ddof = 1) over seeds 0–4.",
         "- **Δ:** setting mean minus reference mean.",
         "- **p-values:** two-sided paired t-test across the 5 seeds (df = 4); `*` marks p < 0.05. "
         "No correction for multiple comparisons.", ""]
    for ds in DATASETS:
        rr = stats_row(ref[ds], ref[ds])
        L += [f"## {ds}", "",
              "| Setting | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | s / epoch |",
              "|---|---|---|---|---|---|---|---|---|",
              f"| **reference** | {rr['acc_mean']:.2f} ± {rr['acc_std']:.2f} | — | — | "
              f"{rr['f1_mean']:.2f} ± {rr['f1_std']:.2f} | — | — | {rr['best_epoch']:.1f} | "
              f"{rr['epoch_time_s']:.2f} |"]
        for _, setting, _ in N1_SETTINGS:
            r = rows.get((setting, ds))
            if r is None:
                L.append(f"| {LABELS[setting]} | FAILED / missing | | | | | | | |")
                continue
            L.append(f"| {LABELS[setting]} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | "
                     f"{r['d_acc']:+.2f} | {fmt_p(r['p_acc'])} | {r['f1_mean']:.2f} ± {r['f1_std']:.2f} | "
                     f"{r['d_f1']:+.2f} | {fmt_p(r['p_f1'])} | {r['best_epoch']:.1f} | "
                     f"{r['epoch_time_s']:.2f} |")
        L += ["", "Per-seed accuracy (%), seeds 0–4: reference " + rr["per_seed_acc"] + "; " +
              "; ".join(f"{setting} {rows[(setting, ds)]['per_seed_acc']}"
                        for _, setting, _ in N1_SETTINGS if (setting, ds) in rows), ""]

    # learned beta
    L += ["## Learned β at the best checkpoint", "",
          "Mean ± sample std over the 5 seeds. Rows are layers, columns are heads.", ""]
    for (setting, ds), runs in runs_by.items():
        beta = np.array([r["attn_beta"] for r in runs])          # (seeds, layers, heads)
        mu, sd = beta.mean(0), beta.std(0, ddof=1)
        L += [f"**{LABELS[setting]}, {ds}**", "",
              "| Layer | " + " | ".join(f"head {h}" for h in range(beta.shape[2])) + " |",
              "|---|" + "---|" * beta.shape[2]]
        for l in range(beta.shape[1]):
            L.append(f"| {l} | " + " | ".join(f"{mu[l, h]:+.3f} ± {sd[l, h]:.3f}"
                                              for h in range(beta.shape[2])) + " |")
        L.append("")

    # preprocessing cost
    L += ["## Cost of building B (k = 23, intimacy sampler)", "",
          "| Dataset | Version | Graph rebuild (s) | Raw scores (s) | Normalise (s) | Total (s) | Storage (MB, float32) |",
          "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for _, setting, _ in N1_SETTINGS:
            if (setting, ds) not in runs_by:
                continue
            i = runs_by[(setting, ds)][0]["attn_bias_build"]
            L.append(f"| {ds} | {i['version']} | {i['graph_rebuild_s']:.1f} | {i['raw_scores_s']:.1f} | "
                     f"{i['normalize_s']:.2f} | {i['total_build_s']:.1f} | {i['storage_mb']:.1f} |")
    L += ["", "Raw intimacy scores need the same dense factorisation of the n × n graph matrix as "
          "the main preprocessing. Edge weights are read straight from the sparse graph. B is "
          "built once per dataset and cached.", ""]

    # optional attention analysis
    apath = os.path.join(OUT, "attention_by_rank.json")
    if os.path.exists(apath):
        a = json.load(open(apath))
        L += ["## Target-to-helper attention by intimacy rank (seed 0, test set)", "",
              "Average attention weight from the target position to one helper in each rank "
              "group, averaged over test documents, heads and both layers. For comparison, "
              "uniform attention over the 24 positions would be 0.042.", "",
              "| Dataset | Setting | Self | Ranks 1–5 | Ranks 6–12 | Ranks 13–23 | Seed-0 accuracy reproduced |",
              "|---|---|---|---|---|---|---|"]
        for ds, d in a.items():
            for name, r in d.items():
                L.append(f"| {ds} | {name} | {r['self']:.3f} | {r['mean_per_helper_1-5']:.3f} | "
                         f"{r['mean_per_helper_6-12']:.3f} | {r['mean_per_helper_13-23']:.3f} | "
                         f"{'yes' if r['reproduced'] else 'NO'} |")
        L.append("")

    log_path = "results/experiments/run_log.json"
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    failed = [f"{k} (exit {v['exit_code']}, see {v['log']})" for k, v in log.items()
              if k.startswith("N1/") and not v["complete"]]
    L += ["## Failures", ""]
    L += [f"- {x}" for x in failed + missing] if (failed or missing) else \
        [f"None: all {len(N1_SETTINGS)} settings × {len(DATASETS)} datasets × {len(SEEDS)} seeds completed."]
    L.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
