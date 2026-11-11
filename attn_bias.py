"""N1 edge-aware attention: per-slice graph-score matrices B.

For each document's input slice [target, helper_1 .. helper_k] we build one
(k+1) x (k+1) matrix B that is added, scaled by a learnable beta[l, h], to the
attention scores (see ``model.MultiHeadSelfAttention``).

Two versions:
  * ``intimacy`` (N1a): raw = S[i, j] from paper Eq. 2, transformed log(raw + 1e-10).
  * ``edge``     (N1b): raw = edge weight of the graph without self-loops (TF-IDF for
    document-word pairs, positive PMI for word-word pairs, 0 if no edge),
    transformed log(1 + raw).

Both are then z-scored within each slice over the off-diagonal entries between real
(non-padded) positions, using the population std (ddof = 0); if that std is below
1e-8 all off-diagonal entries become 0. The diagonal and every row/column of a
padded position are set to 0 (padded keys are still removed by the attention mask).

B is cached per dataset in ``data/processed/<ds>/`` as float32, together with the
slice node ids it was built for and the build time.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.linalg as sla

import graph as graph_mod
from data import build_vocab, load_corpus
from sampling import build_slices

INTIMACY_EPS = 1e-10
STD_EPS = 1e-8


def normalize_bias(raw: np.ndarray, mask: np.ndarray, version: str) -> np.ndarray:
    """Transform and z-score raw slice scores into B.

    Args:
        raw: (N, L, L) raw pairwise scores (entries at padded positions are ignored).
        mask: (N, L) bool, True for real positions.
        version: ``intimacy`` -> log(raw + 1e-10); ``edge`` -> log(1 + raw).

    Returns:
        (N, L, L) float32 B: zero diagonal, zero at padded rows/columns, and per slice
        off-diagonal real entries with mean 0 and population std 1 (or all 0 if the
        std is < 1e-8).
    """
    raw = np.asarray(raw, dtype=np.float64)
    N, L, _ = raw.shape
    if version == "intimacy":
        val = np.log(np.maximum(raw, 0.0) + INTIMACY_EPS)
    elif version == "edge":
        val = np.log1p(raw)
    else:
        raise ValueError(f"Unknown attention bias version '{version}'")
    valid = mask[:, :, None] & mask[:, None, :] & ~np.eye(L, dtype=bool)[None]
    cnt = valid.sum(axis=(1, 2))
    val = np.where(valid, val, 0.0)
    mean = val.sum(axis=(1, 2)) / np.maximum(cnt, 1)
    centred = np.where(valid, val - mean[:, None, None], 0.0)
    std = np.sqrt((centred ** 2).sum(axis=(1, 2)) / np.maximum(cnt, 1))
    ok = std >= STD_EPS
    B = np.where(ok[:, None, None], centred / np.where(ok, std, 1.0)[:, None, None], 0.0)
    return np.where(valid, B, 0.0).astype(np.float32)


def _pair_index(node_ids: np.ndarray, mask: np.ndarray):
    """Flattened (u, v) node pairs of all slices, with -1 where either side is padding."""
    N, L = node_ids.shape
    real = mask[:, :, None] & mask[:, None, :]
    U = np.where(real, np.broadcast_to(node_ids[:, :, None], (N, L, L)), -1).ravel()
    V = np.where(real, np.broadcast_to(node_ids[:, None, :], (N, L, L)), -1).ravel()
    return U, V


def raw_intimacy(A, n_docs: int, node_ids, mask, alpha: float, block_size: int):
    """S[u, v] (paper Eq. 2) for every pair in every slice, without keeping S.

    Same factorisation as ``graph.intimacy_and_helpers``: Cholesky of
    M = I - (1 - alpha) * Abar, then column blocks of S by triangular solves. From
    each block only the entries needed by the slices are kept.

    Returns:
        (N, L, L) float64 raw intimacy (0 at padded pairs).
    """
    n = A.shape[0]
    N, L = node_ids.shape
    U, V = _pair_index(node_ids, mask)
    out = np.zeros(U.shape[0], np.float64)
    keep = np.flatnonzero(V >= 0)
    keep = keep[np.argsort(V[keep], kind="stable")]
    Vs = V[keep]

    M = graph_mod.normalized_adjacency(A).astype(np.float32).toarray()
    M *= -(1.0 - alpha)
    M[np.diag_indices(n)] += 1.0
    cf = sla.cho_factor(M.T, lower=True, overwrite_a=True, check_finite=False)
    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        lo, hi = np.searchsorted(Vs, [s, e])
        if lo == hi:
            continue
        E = np.zeros((n, e - s), np.float32)
        E[np.arange(s, e), np.arange(e - s)] = 1.0
        Sb = sla.cho_solve(cf, E, overwrite_b=True, check_finite=False)
        Sb *= alpha
        idx = keep[lo:hi]
        out[idx] = Sb[U[idx], V[idx] - s]       # column v of the block = S[:, v]
        del Sb
    del cf, M
    out = out.reshape(N, L, L)
    # S is symmetric in exact arithmetic; the float32 solves differ from symmetry by
    # rounding (~1e-6 relative), so symmetrise to make B exactly symmetric.
    return 0.5 * (out + out.transpose(0, 2, 1))


def raw_edge(A, node_ids, mask):
    """Edge weight A[u, v] (TF-IDF or positive PMI; 0 if no edge) for every slice pair.

    Self-loops are ignored: the diagonal of every slice is zeroed later anyway, and
    a slice never contains the same node twice at different positions.

    Returns:
        (N, L, L) float64 raw edge weights (0 at padded pairs).
    """
    N, L = node_ids.shape
    U, V = _pair_index(node_ids, mask)
    out = np.zeros(U.shape[0], np.float64)
    ok = (U >= 0) & (U != V)
    out[ok] = np.asarray(A[U[ok], V[ok]]).ravel()
    return out.reshape(N, L, L)


def rebuild_graph(data, cfg):
    """Rebuild the text graph A (with self-loops) from the raw corpus, as in preprocessing.

    Checks that the rebuilt vocabulary and document-word TF-IDF match the cache.
    """
    docs, _, is_test = load_corpus(cfg["paths"]["textgcn_dir"], data.name)
    vocab = build_vocab(docs)
    if len(vocab) != data.n_words or len(docs) != data.n_docs:
        raise RuntimeError("rebuilt corpus does not match the preprocessing cache")
    w2i = {w: i for i, w in enumerate(vocab)}
    docs_idx = [np.array([w2i[w] for w in d], dtype=np.int64) for d in docs]
    A, tfidf, _ = graph_mod.build_text_graph(docs_idx, data.n_words,
                                             cfg["preprocess"]["window_size"])
    if not (np.array_equal(is_test, data.is_test) and (tfidf != data.tfidf).nnz == 0):
        raise RuntimeError("rebuilt graph does not match the preprocessing cache")
    return A


def cache_path(processed_dir: str, name: str, version: str, sampler: str, k: int, seed: int):
    """Cache file of B; the seed is part of the key only for the random sampler."""
    tag = f"attn_bias_{version}_{sampler}_k{k}" + (f"_seed{seed}" if sampler == "random" else "")
    return os.path.join(processed_dir, name, tag + ".npz")


def get_attn_bias(data, cfg: dict, seed: int = 0, force: bool = False) -> dict:
    """Return (building and caching if needed) B for the configured sampler, k and version.

    Returns:
        dict with ``B`` (n_docs, k+1, k+1) float32, ``node_ids`` (n_docs, k+1) of the
        slices B was built for, and ``info`` (version, build time, storage size, ...).
    """
    m = cfg["model"]
    version = m.get("attn_bias", "none")
    if version == "none":
        raise ValueError("attn_bias is 'none'")
    path = cache_path(cfg["paths"]["processed_dir"], data.name, version, m["sampler"], m["k"], seed)
    node_ids, mask, _ = build_slices(data, m["sampler"], m["k"], seed)
    if not force and os.path.exists(path):
        z = np.load(path)
        if np.array_equal(z["node_ids"], node_ids):
            info = json.loads(str(z["info"]))
            print(f"[{data.name}] loaded attention bias B from {path}")
            return {"B": z["B"], "node_ids": node_ids, "info": info}
        print(f"[{data.name}] cached B is for different slices; rebuilding")

    print(f"[{data.name}] building attention bias B ({version}, sampler={m['sampler']}, "
          f"k={m['k']})", flush=True)
    t0 = time.perf_counter()
    A = rebuild_graph(data, cfg)
    t_graph = time.perf_counter() - t0
    t1 = time.perf_counter()
    if version == "intimacy":
        p = cfg["preprocess"]
        raw = raw_intimacy(A, data.n_docs, node_ids, mask, p["alpha"], p["s_block_size"])
    else:
        raw = raw_edge(A, node_ids, mask)
    del A
    t_raw = time.perf_counter() - t1
    t2 = time.perf_counter()
    B = normalize_bias(raw, mask, version)
    t_norm = time.perf_counter() - t2
    check = {}
    if version == "intimacy" and m["sampler"] == "intimacy":
        # the recomputed S must reproduce the cached helper scores (row 0 = target)
        ref = data.helper_scores[:, :m["k"]].astype(np.float64)
        check["max_abs_diff_vs_cached_helper_scores"] = float(np.abs(raw[:, 0, 1:] - ref).max())
    info = {
        "version": version, "sampler": m["sampler"], "k": m["k"],
        "shape": list(B.shape), "dtype": "float32", "storage_mb": B.nbytes / 2 ** 20,
        "graph_rebuild_s": t_graph, "raw_scores_s": t_raw, "normalize_s": t_norm,
        "total_build_s": time.perf_counter() - t0,
        "n_constant_slices": int(np.sum(np.all(B == 0, axis=(1, 2)))), **check,
    }
    np.savez(path, B=B, node_ids=node_ids, info=json.dumps(info))
    print(f"[{data.name}] B built in {info['total_build_s']:.1f}s, "
          f"{info['storage_mb']:.1f} MB, {info['n_constant_slices']} all-zero slices "
          f"{check}", flush=True)
    return {"B": B, "node_ids": node_ids, "info": info}
