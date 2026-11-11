"""Data loading and the cached preprocessing pipeline for TG-Transformer.

This module

* reads the cleaned TextGCN corpora (``<prefix>.clean.txt``) and their split/label
  files (``<prefix>.txt``),
* builds the vocabulary and the GloVe-based node feature table X,
* orchestrates the one-off preprocessing (graph -> intimacy S -> helper lists ->
  WL codes), prints all sanity checks, and caches everything to disk.

Only the artefacts needed for training are cached (node types, X, labels, split,
top-50 helper lists, WL codes, and the document-word TF-IDF matrix used by the
``tfidf`` sampler). The adjacency matrix A and the intimacy matrix S are freed
after use.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import graph as graph_mod
import wl as wl_mod

# Canonical dataset names -> file prefix used by the TextGCN repository.
DATASET_PREFIX = {"R8": "R8", "Ohsumed": "ohsumed"}

# Paper Table 1 (train, test, vocab, classes, avg words per document).
PAPER_TABLE1 = {
    "R8": {"train": 5485, "test": 2189, "vocab": 7688, "classes": 8, "avg_words": 65.72},
    "Ohsumed": {"train": 3357, "test": 4043, "vocab": 14157, "classes": 23, "avg_words": 135.82},
}


def canonical_name(name: str) -> str:
    """Map a user-supplied dataset name (case-insensitive) to its canonical form."""
    for canon in DATASET_PREFIX:
        if canon.lower() == name.lower():
            return canon
    raise ValueError(f"Unknown dataset '{name}'. Known: {list(DATASET_PREFIX)}")


# --------------------------------------------------------------------------- #
# Raw corpus and features
# --------------------------------------------------------------------------- #
def load_corpus(raw_dir: str, name: str):
    """Load a cleaned TextGCN corpus.

    Args:
        raw_dir: directory holding ``<prefix>.txt`` (meta) and ``<prefix>.clean.txt``.
        name: canonical dataset name.

    Returns:
        docs: list of token lists (one per document, corpus order).
        labels: list of label strings.
        is_test: bool array, True for test documents. The meta files mark training
            documents as ``train`` (R8) or ``training`` (Ohsumed).
    """
    prefix = DATASET_PREFIX[name]
    with open(os.path.join(raw_dir, f"{prefix}.txt"), encoding="latin1") as f:
        meta = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    with open(os.path.join(raw_dir, f"{prefix}.clean.txt"), encoding="latin1") as f:
        docs = [line.split() for line in f.read().split("\n")]
    if len(docs) == len(meta) + 1 and not docs[-1]:
        docs = docs[:-1]
    if len(docs) != len(meta):
        raise ValueError(f"{name}: {len(docs)} documents but {len(meta)} meta lines")
    splits = [m[1] for m in meta]
    labels = [m[2] for m in meta]
    unknown = set(splits) - {"train", "training", "test"}
    if unknown:
        raise ValueError(f"{name}: unexpected split markers {unknown}")
    is_test = np.array([s == "test" for s in splits], dtype=bool)
    return docs, labels, is_test


def build_vocab(docs):
    """Return the sorted vocabulary of all documents (train and test, TextGCN convention)."""
    return sorted({w for d in docs for w in d})


def load_glove(path: str, vocab, dim: int = 300):
    """Read GloVe vectors for the words in ``vocab``.

    Returns:
        dict word -> float32 vector, only for vocabulary words present in GloVe.
    """
    wanted = set(vocab)
    vecs = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word, rest = line.split(" ", 1)
            if word in wanted:
                v = np.asarray(rest.split(), dtype=np.float32)
                if v.shape[0] == dim:
                    vecs[word] = v
    return vecs


def build_word_table(vocab, glove: dict, dim: int, oov_seed: int, oov_range: float):
    """Build the word feature table: GloVe vector, or uniform(-r, r) for OOV words.

    OOV vectors are drawn in vocabulary order from ``RandomState(oov_seed)``.

    Returns:
        table: (n_words, dim) float32.
        in_glove: (n_words,) bool mask of words found in GloVe.
    """
    rng = np.random.RandomState(oov_seed)
    table = np.empty((len(vocab), dim), dtype=np.float32)
    in_glove = np.zeros(len(vocab), dtype=bool)
    for i, w in enumerate(vocab):
        if w in glove:
            table[i] = glove[w]
            in_glove[i] = True
        else:
            table[i] = rng.uniform(-oov_range, oov_range, dim).astype(np.float32)
    return table, in_glove


def build_features(docs_idx, word_table):
    """Build the node feature matrix X (documents first, then words).

    Word rows are the word table. A document's row is the mean of the word-table
    rows over all of its tokens (repeated tokens count each time). Empty documents
    get a zero vector.

    Args:
        docs_idx: list of int arrays (word ids per document).
        word_table: (n_words, d) float32.

    Returns:
        X: (n_docs + n_words, d) float32.
    """
    n_docs = len(docs_idx)
    n_words, d = word_table.shape
    X = np.zeros((n_docs + n_words, d), dtype=np.float32)
    for i, ids in enumerate(docs_idx):
        if len(ids):
            X[i] = word_table[ids].mean(axis=0)
    X[n_docs:] = word_table
    return X


# --------------------------------------------------------------------------- #
# Cached preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class PreparedData:
    """All cached artefacts for one dataset (everything training needs)."""

    name: str
    n_docs: int
    n_words: int
    node_types: np.ndarray      # (n_nodes,) int64, 0 = document, 1 = word
    X: np.ndarray               # (n_nodes, d) float32
    labels: np.ndarray          # (n_docs,) int64; test entries used ONLY in final evaluation
    is_test: np.ndarray         # (n_docs,) bool
    helpers: np.ndarray         # (n_docs, 50) int64 global node ids of word nodes
    helper_scores: np.ndarray   # (n_docs, 50) float32 intimacy scores
    wl_codes: np.ndarray        # (n_nodes,) int64
    tfidf: sp.csr_matrix        # (n_docs, n_words) document-word TF-IDF weights
    classes: list

    @property
    def n_nodes(self) -> int:
        """Total number of graph nodes."""
        return self.n_docs + self.n_words

    @property
    def num_classes(self) -> int:
        """Number of target classes."""
        return len(self.classes)


def _cache_paths(processed_dir: str, name: str):
    """Return (dir, arrays.npz, tfidf.npz, meta.json) cache paths for a dataset."""
    d = os.path.join(processed_dir, name)
    return d, os.path.join(d, "arrays.npz"), os.path.join(d, "tfidf.npz"), os.path.join(d, "meta.json")


def load_prepared(processed_dir: str, name: str) -> PreparedData:
    """Load cached preprocessing artefacts from disk."""
    _, arr_p, tfidf_p, meta_p = _cache_paths(processed_dir, name)
    arr = np.load(arr_p)
    with open(meta_p) as f:
        meta = json.load(f)
    return PreparedData(
        name=name, n_docs=meta["n_docs"], n_words=meta["n_words"],
        node_types=arr["node_types"], X=arr["X"], labels=arr["labels"],
        is_test=arr["is_test"], helpers=arr["helpers"], helper_scores=arr["helper_scores"],
        wl_codes=arr["wl_codes"], tfidf=sp.load_npz(tfidf_p).tocsr(), classes=meta["classes"],
    )


def prepare_dataset(name: str, cfg: dict, force: bool = False) -> PreparedData:
    """Run (or load from cache) the full preprocessing pipeline for one dataset.

    Steps: load corpus and verify paper Table 1 statistics;
    GloVe features; text graph (paper Sec 2.1, Eq. 1); intimacy matrix S (paper
    Eq. 2) and top-50 word helpers per document (paper Sec 2.2); WL codes (paper
    Sec 2.3). Prints every sanity check and writes
    ``<results_dir>/preprocessing_<name>.json`` with statistics and timings.
    """
    name = canonical_name(name)
    paths, pcfg = cfg["paths"], cfg["preprocess"]
    out_dir, arr_p, tfidf_p, meta_p = _cache_paths(paths["processed_dir"], name)
    if not force and all(os.path.exists(p) for p in (arr_p, tfidf_p, meta_p)):
        print(f"[{name}] loading cached preprocessing from {out_dir}")
        return load_prepared(paths["processed_dir"], name)

    os.makedirs(out_dir, exist_ok=True)
    timing, report = {}, {"dataset": name}
    print(f"\n========== Preprocessing {name} ==========")

    # ---- 0. corpus & Table 1 statistics -------------------------------------
    t0 = time.perf_counter()
    docs, label_str, is_test = load_corpus(paths["textgcn_dir"], name)
    vocab = build_vocab(docs)
    w2i = {w: i for i, w in enumerate(vocab)}
    docs_idx = [np.array([w2i[w] for w in d], dtype=np.int64) for d in docs]
    classes = sorted(set(label_str))
    c2i = {c: i for i, c in enumerate(classes)}
    labels = np.array([c2i[c] for c in label_str], dtype=np.int64)
    n_docs, n_words = len(docs), len(vocab)
    del docs
    stats = {
        "train": int((~is_test).sum()), "test": int(is_test.sum()), "vocab": n_words,
        "classes": len(classes),
        "avg_words": round(float(np.mean([len(d) for d in docs_idx])), 2),
        "empty_docs": int(sum(len(d) == 0 for d in docs_idx)),
    }
    paper = PAPER_TABLE1[name]
    report["stats"] = stats
    report["paper_table1"] = paper
    report["stats_deviation"] = {k: round(stats[k] - paper[k], 4) for k in paper}
    timing["load_corpus_s"] = time.perf_counter() - t0
    print(f"[{name}] statistics (ours vs paper Table 1):")
    for k in paper:
        tag = "match" if stats[k] == paper[k] else f"diff {stats[k] - paper[k]:+g}"
        print(f"    {k:>10}: {stats[k]:>9}   paper {paper[k]:>9}   [{tag}]")
    if stats["empty_docs"]:
        print(f"    WARNING: {stats['empty_docs']} empty documents")

    # ---- 1. GloVe features --------------------------------------------------
    t0 = time.perf_counter()
    glove = load_glove(paths["glove"], vocab, pcfg["emb_dim"])
    word_table, in_glove = build_word_table(vocab, glove, pcfg["emb_dim"],
                                            pcfg["oov_seed"], pcfg["oov_range"])
    del glove
    X = build_features(docs_idx, word_table)
    del word_table
    token_counts = np.bincount(np.concatenate(docs_idx), minlength=n_words)
    report["glove"] = {
        "type_coverage": float(in_glove.mean()),
        "token_coverage": float(token_counts[in_glove].sum() / token_counts.sum()),
        "oov_types": int((~in_glove).sum()),
    }
    timing["features_s"] = time.perf_counter() - t0
    print(f"[{name}] GloVe coverage: {in_glove.sum()}/{n_words} word types "
          f"({100 * report['glove']['type_coverage']:.2f}%), "
          f"{100 * report['glove']['token_coverage']:.2f}% of tokens")

    # ---- 2. graph (paper Sec 2.1) -------------------------------------------
    t0 = time.perf_counter()
    A, tfidf, gstats = graph_mod.build_text_graph(docs_idx, n_words, pcfg["window_size"])
    timing["graph_build_s"] = time.perf_counter() - t0
    report["graph"] = gstats
    print(f"[{name}] graph: {gstats['n_nodes']} nodes, {gstats['doc_word_edges']} doc-word edges, "
          f"{gstats['word_word_edges']} word-word edges (PMI>0) "
          f"[{timing['graph_build_s']:.1f}s]")
    node_types = np.concatenate([np.zeros(n_docs, np.int64), np.ones(n_words, np.int64)])

    # ---- 3. WL codes (paper Sec 2.3) — on the unweighted graph --------------
    t0 = time.perf_counter()
    wl_codes, wl_info = wl_mod.wl_codes(A, max_iter=pcfg["wl_max_iter"])
    timing["wl_s"] = time.perf_counter() - t0
    report["wl"] = wl_info
    print(f"[{name}] WL: {wl_info['n_colors']} colors after {wl_info['iterations']} iterations, "
          f"converged={wl_info['converged']}, history={wl_info['colors_per_iter']} "
          f"[{timing['wl_s']:.1f}s]")

    # ---- 4. intimacy S (paper Eq. 2) + top-50 helpers ------------------------
    S_res = graph_mod.intimacy_and_helpers(
        A, n_docs, alpha=pcfg["alpha"], top_n=pcfg["top_helpers"],
        block_size=pcfg["s_block_size"])
    del A
    gc.collect()
    timing["S_compute_s"] = S_res["time_S_s"]
    timing["helper_selection_s"] = S_res["time_helpers_s"]
    helpers, helper_scores = S_res["helpers"], S_res["helper_scores"]
    report["S"] = S_res["stats"]
    st = S_res["stats"]
    print(f"[{name}] S computed in {timing['S_compute_s']:.1f}s "
          f"(helper selection {timing['helper_selection_s']:.1f}s)")
    print(f"    row min  : min {st['row_min']['min']:.3e}  mean {st['row_min']['mean']:.3e}  "
          f"max {st['row_min']['max']:.3e}")
    print(f"    row max  : min {st['row_max']['min']:.3e}  mean {st['row_max']['mean']:.3e}  "
          f"max {st['row_max']['max']:.3e}")
    print(f"    row sum  : min {st['row_sum']['min']:.4f}  mean {st['row_sum']['mean']:.4f}  "
          f"max {st['row_sum']['max']:.4f}")
    print(f"    diagonal is the row maximum in {st['diag_is_row_max']}/{st['n_rows']} rows "
          f"(strictly largest in {st['diag_strictly_largest']})")

    # ---- 5. helper sanity checks --------------------------------------------
    only_words = bool(np.all(node_types[helpers] == 1))
    no_self = bool(np.all(helpers != np.arange(n_docs)[:, None]))
    unique_rows = bool(all(len(set(r)) == len(r) for r in helpers))
    sorted_desc = bool(np.all(np.diff(helper_scores, axis=1) <= 0))
    report["helpers"] = {
        "shape": list(helpers.shape), "only_word_nodes": only_words, "no_self": no_self,
        "unique_per_row": unique_rows, "sorted_by_intimacy_desc": sorted_desc,
        "frac_helpers_in_own_doc": float(np.mean([
            np.isin(helpers[i] - n_docs, docs_idx[i]).mean() for i in range(n_docs)])),
    }
    print(f"[{name}] helpers: shape {helpers.shape}, only word nodes={only_words}, "
          f"no self={no_self}, unique={unique_rows}, sorted desc={sorted_desc}, "
          f"{100 * report['helpers']['frac_helpers_in_own_doc']:.1f}% of helpers occur in "
          f"their own document")
    assert helpers.shape == (n_docs, pcfg["top_helpers"]) and only_words and no_self

    # ---- 6. save ------------------------------------------------------------
    np.savez(arr_p, node_types=node_types, X=X, labels=labels, is_test=is_test,
             helpers=helpers, helper_scores=helper_scores, wl_codes=wl_codes)
    sp.save_npz(tfidf_p, tfidf)
    with open(meta_p, "w") as f:
        json.dump({"n_docs": n_docs, "n_words": n_words, "classes": classes}, f)
    with open(os.path.join(out_dir, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))

    report["timing_s"] = {k: round(v, 3) for k, v in timing.items()}
    os.makedirs(paths["results_dir"], exist_ok=True)
    with open(os.path.join(paths["results_dir"], f"preprocessing_{name}.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[{name}] timings (s): {report['timing_s']}")
    return load_prepared(paths["processed_dir"], name)
