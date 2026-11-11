"""Summarise experiment W1 (WL computed inside each sampled slice).

Three rows per dataset: the baseline (full-graph WL), W1 (subgraph WL) and E1
"type on, WL off" (no WL at all, reused from results/experiments/e1/). Same
statistics as ``summarize_experiments.py`` (mean +/- sample std over seeds 0-4,
difference from the baseline, two-sided paired t-test across seeds), plus a paired
t-test of W1 against "WL off", WL code statistics for both scopes, and the
preprocessing time of subgraph WL.

Outputs: results/experiments/W1/summary.md, summary.csv
"""

import csv
import json
import os

import numpy as np
from scipy.stats import ttest_rel

from run_experiments import DATASETS, SEEDS, W1_SETTINGS
from summarize_experiments import fmt_p, load_runs, stats_row

ROOT = os.path.dirname(os.path.abspath(__file__))
REF_DIR = "results/baseline"
NOWL_DIR = "results/experiments/e1/type_on_wl_off"
OUT = "results/experiments/W1"
W1_DIR = os.path.join(OUT, W1_SETTINGS[0][1])


def paired(a_runs, b_runs, key):
    """Mean difference (a - b, in %) and two-sided paired t-test p-value."""
    a = np.array([r[key] for r in a_runs]) * 100
    b = np.array([r[key] for r in b_runs]) * 100
    return a.mean() - b.mean(), ttest_rel(a, b).pvalue


def main():
    """Build summary.md / summary.csv for W1."""
    os.chdir(ROOT)
    rows, w1_runs, missing = {}, {}, []
    for ds in DATASETS:
        ref, nowl, w1 = load_runs(REF_DIR, ds), load_runs(NOWL_DIR, ds), load_runs(W1_DIR, ds)
        rows[(ds, "baseline")] = stats_row(ref, ref)
        rows[(ds, "nowl")] = stats_row(nowl, ref)
        if any(r is None for r in w1):
            missing.append(f"W1/{ds} ({sum(r is None for r in w1)} seeds missing)")
            continue
        rows[(ds, "w1")] = stats_row(w1, ref)
        w1_runs[ds] = (w1, nowl)

    names = [("baseline", "baseline (full-graph WL)"), ("w1", "W1: subgraph WL"),
             ("nowl", "E1: type on, WL off")]
    fields = ["dataset", "setting", "acc_mean", "acc_std", "d_acc", "p_acc", "f1_mean", "f1_std",
              "d_f1", "p_f1", "best_epoch", "epoch_time_s", "per_seed_acc"]
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ds in DATASETS:
            for key, _ in names:
                if (ds, key) in rows:
                    r = rows[(ds, key)]
                    w.writerow({"dataset": ds, "setting": key,
                                **{k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in r.items() if k in fields}})

    L = ["# W1: WL computed inside each sampled slice", "",
         "The baseline computes WL codes once on the full text graph. W1 follows the paper's "
         "wording more literally: WL refinement runs inside each document's 24-node slice "
         "(the induced subgraph, unweighted, no self-loops), with one code dictionary shared "
         "by all slices. Only the WL codes change; the Eq. 5 encoding and everything else are "
         "as in the baseline.", "",
         f"Reference: `configs/baseline.yaml`, results reused from `{REF_DIR}/`. The \"WL off\" "
         f"row is E1 \"type on, WL off\", reused from `{NOWL_DIR}/`.", "",
         "- **Accuracy and macro-F1:** in %, mean ± sample std (ddof = 1) over seeds 0–4.",
         "- **Δ:** setting mean minus baseline mean.",
         "- **p-values:** two-sided paired t-test across the 5 seeds (df = 4); `*` marks p < 0.05. "
         "No correction for multiple comparisons.", ""]
    for ds in DATASETS:
        L += [f"## {ds}", "",
              "| Setting | Accuracy | Δ acc | p (acc) | Macro-F1 | Δ F1 | p (F1) | Best epoch | s / epoch |",
              "|---|---|---|---|---|---|---|---|---|"]
        for key, label in names:
            r = rows.get((ds, key))
            if r is None:
                L.append(f"| {label} | FAILED / missing | | | | | | | |")
            elif key == "baseline":
                L.append(f"| **{label}** | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | — | — | "
                         f"{r['f1_mean']:.2f} ± {r['f1_std']:.2f} | — | — | {r['best_epoch']:.1f} | "
                         f"{r['epoch_time_s']:.2f} |")
            else:
                L.append(f"| {label} | {r['acc_mean']:.2f} ± {r['acc_std']:.2f} | {r['d_acc']:+.2f} | "
                         f"{fmt_p(r['p_acc'])} | {r['f1_mean']:.2f} ± {r['f1_std']:.2f} | "
                         f"{r['d_f1']:+.2f} | {fmt_p(r['p_f1'])} | {r['best_epoch']:.1f} | "
                         f"{r['epoch_time_s']:.2f} |")
        L += ["", "Per-seed accuracy (%), seeds 0–4: " + "; ".join(
            f"{key} {rows[(ds, key)]['per_seed_acc']}" for key, _ in names if (ds, key) in rows), ""]

    L += ["## Does subgraph WL beat no WL at all?", "",
          "Paired t-test of W1 against E1 \"type on, WL off\" (W1 minus WL off).", "",
          "| Dataset | Δ acc | p (acc) | Δ F1 | p (F1) |", "|---|---|---|---|---|"]
    for ds, (w1, nowl) in w1_runs.items():
        da, pa = paired(w1, nowl, "test_acc")
        df, pf = paired(w1, nowl, "test_macro_f1")
        L.append(f"| {ds} | {da:+.2f} | {fmt_p(pa)} | {df:+.2f} | {fmt_p(pf)} |")
    L.append("")

    # WL statistics and preprocessing time (stored in every W1 result file)
    info = {ds: w1_runs[ds][0][0]["wl_subgraph_build"] for ds in w1_runs}
    if info:
        L += ["## WL code statistics", "",
              "Counted over all (slice, position) entries: 24 per document, all documents. "
              "\"Unique\" means the code occurs in exactly one entry. For full-graph WL a word "
              "keeps one code in every slice it appears in.", "",
              "| Dataset | Scope | Entries | Distinct codes | Min code | Max code | Entries with a unique code |",
              "|---|---|---|---|---|---|---|"]
        for ds, i in info.items():
            for scope in ("full", "subgraph"):
                s = i[f"stats_{scope}"]
                L.append(f"| {ds} | {scope} | {s['n_entries']:,} | {s['n_distinct_codes']:,} | "
                         f"{s['min_code']} | {s['max_code']} | "
                         f"{100 * s['frac_entries_with_unique_code']:.1f}% |")
        L += ["", "Most common codes (code: count):", ""]
        for ds, i in info.items():
            for scope in ("full", "subgraph"):
                top = ", ".join(f"{c}: {n}" for c, n in i[f"stats_{scope}"]["most_common"])
                L.append(f"- **{ds}, {scope}:** {top}")
        L += ["", "## Subgraph WL: refinement and cost", "",
              "| Dataset | Rounds | Codes after each round | Edges per slice (min / mean / max) | "
              "Graph rebuild (s) | Subgraphs (s) | WL (s) | Total (s) |",
              "|---|---|---|---|---|---|---|---|"]
        for ds, i in info.items():
            e = i["edges_per_slice"]
            L.append(f"| {ds} | {i['wl']['rounds']} | {' → '.join(f'{c:,}' for c in i['wl']['codes_per_round'])} | "
                     f"{e['min']} / {e['mean']:.0f} / {e['max']} | {i['graph_rebuild_s']:.1f} | "
                     f"{i['subgraph_extract_s']:.1f} | {i['wl_s']:.1f} | {i['total_build_s']:.1f} |")
        L += ["", "All slices are refined together with one dictionary per round, so codes are "
              "comparable across slices; the loop stops once no slice's partition changes. Each "
              "slice ends with the same partition as WL run on that slice alone (checked in the "
              "tests).", ""]

    log_path = "results/experiments/run_log.json"
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    failed = [f"{k} (exit {v['exit_code']}, see {v['log']})" for k, v in log.items()
              if k.startswith("W1/") and not v["complete"]]
    L += ["## Failures", ""]
    L += [f"- {x}" for x in failed + missing] if (failed or missing) else \
        [f"None: all {len(DATASETS)} datasets × {len(SEEDS)} seeds completed."]
    L.append("")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
