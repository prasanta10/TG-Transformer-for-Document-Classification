"""E2: preprocessing time of each sampler (intimacy, tfidf, random), k = 23.

Sampler-specific cost = upstream precomputation that only this sampler needs +
building the (n_docs, k+1) slices.
  * intimacy: text graph incl. PMI (paper Sec 2.1) + intimacy S (Eq. 2) + top-50
    helper selection, taken from results/preprocessing_<ds>.json (measured during
    preprocessing), + slicing the cached top-50 lists (measured here);
  * tfidf: document-word TF-IDF matrix (measured here, from the raw corpus) +
    per-document ranking (measured here);
  * random: drawing k random words per document (measured here).
Note: the WL encoding needs the graph regardless of the sampler.
Times measured here are the median of 3 repetitions.
Output: results/experiments/e2_sampler_timing.json
"""

import json
import os
import statistics
import time

import numpy as np

import graph as G
from data import build_vocab, load_corpus, load_prepared
from sampling import build_slices

ROOT = os.path.dirname(os.path.abspath(__file__))


def med_time(fn, reps=3):
    """Median wall time of ``fn()`` over ``reps`` runs, in seconds."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    """Measure and save per-sampler preprocessing times for R8 and Ohsumed."""
    os.chdir(ROOT)
    out = {}
    for ds in ["R8", "Ohsumed"]:
        pre = json.load(open(f"results/preprocessing_{ds}.json"))["timing_s"]
        d = load_prepared("data/processed", ds)
        docs, _, _ = load_corpus("data/raw/textgcn", ds)
        vocab = build_vocab(docs)
        w2i = {w: i for i, w in enumerate(vocab)}
        di = [np.array([w2i[w] for w in x], np.int64) for x in docs]
        t_tfidf = med_time(lambda: G.doc_word_tfidf(di, len(vocab)))
        slices = {s: med_time(lambda s=s: build_slices(d, s, 23, seed=0))
                  for s in ("intimacy", "tfidf", "random")}
        out[ds] = {
            "intimacy": {"graph_build_s": pre["graph_build_s"], "S_compute_s": pre["S_compute_s"],
                         "helper_selection_s": pre["helper_selection_s"],
                         "slice_build_s": slices["intimacy"]},
            "tfidf": {"tfidf_matrix_s": t_tfidf, "slice_build_s": slices["tfidf"]},
            "random": {"slice_build_s": slices["random"]},
        }
        for s in out[ds]:
            out[ds][s]["total_s"] = round(sum(out[ds][s].values()), 4)
        print(ds, json.dumps(out[ds], indent=1))
    os.makedirs("results/experiments", exist_ok=True)
    with open("results/experiments/e2_sampler_timing.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
