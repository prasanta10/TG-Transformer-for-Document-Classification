"""Run experiments E1-E5.

Each setting changes exactly one thing relative to the reference configuration
(configs/baseline.yaml) and is run on R8 and Ohsumed with seeds 0-4 via run.py,
in its own subprocess. Results go to
results/experiments/<exp>/<setting>/<dataset>/seed_<s>.json (+ curves CSV).
The reference itself is not re-run: its results in results/baseline/ are reused.

Settings that already have all result files are skipped (resumable). A failing
setting (non-zero exit, e.g. error or out-of-memory) is logged and the remaining
settings still run; failures are listed in results/experiments/run_log.json.

Usage:
    python run_experiments.py [--config configs/baseline.yaml] [--only e1 e3]
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["R8", "Ohsumed"]
SEEDS = [0, 1, 2, 3, 4]

# (experiment, setting name, run.py flags) -- one change per setting.
SETTINGS = [
    ("e1", "type_off_wl_off", ["--use_type_enc", "off", "--use_wl_enc", "off"]),
    ("e1", "type_on_wl_off", ["--use_wl_enc", "off"]),
    ("e1", "type_off_wl_on", ["--use_type_enc", "off"]),
    ("e2", "sampler_tfidf", ["--sampler", "tfidf"]),
    ("e2", "sampler_random", ["--sampler", "random"]),
    ("e3", "k5", ["--k", "5"]),
    ("e3", "k10", ["--k", "10"]),
    ("e3", "k15", ["--k", "15"]),
    ("e3", "k30", ["--k", "30"]),
    ("e3", "k40", ["--k", "40"]),
    ("e4", "readout_target", ["--readout", "target"]),
    ("e5", "freeze_on", ["--freeze_embeddings", "on"]),
]


def setting_dir(exp: str, setting: str) -> str:
    """Results directory of one setting, relative to results/."""
    return os.path.join("experiments", exp, setting)


def is_complete(exp: str, setting: str) -> bool:
    """True if every dataset x seed result file of the setting exists."""
    base = os.path.join(ROOT, "results", setting_dir(exp, setting))
    return all(os.path.exists(os.path.join(base, ds, f"seed_{s}.json"))
               for ds in DATASETS for s in SEEDS)


def main():
    """Run all (or selected) settings sequentially and write a run log."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--only", nargs="+", help="restrict to these experiments, e.g. e1 e3")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "logs", "experiments"), exist_ok=True)
    log_path = os.path.join(ROOT, "results", "experiments", "run_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    run_log = json.load(open(log_path)) if os.path.exists(log_path) else {}

    for exp, setting, flags in SETTINGS:
        if args.only and exp not in args.only:
            continue
        key = f"{exp}/{setting}"
        if is_complete(exp, setting):
            print(f"[skip] {key}: already complete", flush=True)
            continue
        cmd = [sys.executable, "run.py", "--config", args.config,
               "--experiment", setting_dir(exp, setting),
               "--datasets", *DATASETS, "--seeds", *map(str, SEEDS), *flags]
        log_file = os.path.join(ROOT, "logs", "experiments", f"{exp}_{setting}.log")
        print(f"[run ] {key}: {' '.join(flags)}", flush=True)
        t0 = time.perf_counter()
        with open(log_file, "w") as f:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        ok = rc == 0 and is_complete(exp, setting)
        run_log[key] = {"flags": flags, "exit_code": rc, "complete": ok,
                        "wall_time_s": round(dt, 1), "log": os.path.relpath(log_file, ROOT)}
        print(f"[{'done' if ok else 'FAIL'}] {key}: exit {rc}, {dt / 60:.1f} min", flush=True)
        with open(log_path, "w") as f:
            json.dump(run_log, f, indent=2)

    failures = [k for k, v in run_log.items() if not v["complete"]]
    print("\nFailures:", failures or "none")


if __name__ == "__main__":
    main()
