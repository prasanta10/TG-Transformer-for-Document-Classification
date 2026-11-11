"""W1: WL codes computed inside each document's sampled slice (paper Sec 2.3, literal reading).

For each slice [target, helper_1 .. helper_k] we take the subgraph of the text graph
induced by its nodes (unweighted, no self-loops) and run WL colour refinement on it
(``wl.wl_codes_batched``, one dictionary shared by all slices). The code of a node
therefore depends on the slice it appears in. Padded positions get code -1 (mapped
to a zero WL vector in the model).

Codes are cached per dataset in ``data/processed/<ds>/`` as (n_docs, k+1) int64,
together with the slice node ids they were built for, the build time and statistics.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter

import numpy as np

from attn_bias import raw_edge, rebuild_graph
from sampling import build_slices
from wl import wl_codes_batched


def slice_adjacency(A, node_ids, mask):
    """Neighbour lists of each slice's induced subgraph (real positions only).

    Returns:
        list (one per slice) of lists of neighbour-position lists.
    """
    W = raw_edge(A, node_ids, mask)            # (N, L, L) edge weights, 0 on the diagonal
    adj = W > 0
    out = []
    for i in range(len(node_ids)):
        real = np.flatnonzero(mask[i])          # real (non-padded) positions
        sub = adj[i][np.ix_(real, real)]
        out.append([np.flatnonzero(row).tolist() for row in sub])
    return out


def code_stats(codes: np.ndarray, mask: np.ndarray, top: int = 10) -> dict:
    """Statistics of WL codes over all real (slice, position) entries."""
    vals = codes[mask]
    cnt = Counter(vals.tolist())
    return {
        "n_entries": int(vals.size), "n_distinct_codes": len(cnt),
        "min_code": int(vals.min()), "max_code": int(vals.max()),
        "frac_entries_with_unique_code": float(sum(1 for v in vals.tolist() if cnt[v] == 1) / vals.size),
        "most_common": [[int(c), int(n)] for c, n in cnt.most_common(top)],
    }


def cache_path(processed_dir: str, name: str, sampler: str, k: int, seed: int):
    """Cache file of the subgraph codes; the seed is part of the key only for the random sampler."""
    tag = f"wl_subgraph_{sampler}_k{k}" + (f"_seed{seed}" if sampler == "random" else "")
    return os.path.join(processed_dir, name, tag + ".npz")


def get_wl_subgraph(data, cfg: dict, seed: int = 0, force: bool = False) -> dict:
    """Return (building and caching if needed) per-slice WL codes for the configured slices.

    Returns:
        dict with ``codes`` (n_docs, k+1) int64 (-1 at padding), ``node_ids`` of the
        slices, and ``info`` (build time, rounds, statistics for both WL scopes).
    """
    m, p = cfg["model"], cfg["preprocess"]
    path = cache_path(cfg["paths"]["processed_dir"], data.name, m["sampler"], m["k"], seed)
    node_ids, mask, _ = build_slices(data, m["sampler"], m["k"], seed)
    if not force and os.path.exists(path):
        z = np.load(path)
        if np.array_equal(z["node_ids"], node_ids):
            print(f"[{data.name}] loaded subgraph WL codes from {path}")
            return {"codes": z["codes"], "node_ids": node_ids, "info": json.loads(str(z["info"]))}
        print(f"[{data.name}] cached subgraph WL codes are for different slices; rebuilding")

    print(f"[{data.name}] computing subgraph WL codes (sampler={m['sampler']}, k={m['k']})",
          flush=True)
    t0 = time.perf_counter()
    A = rebuild_graph(data, cfg)
    t_graph = time.perf_counter() - t0
    t1 = time.perf_counter()
    adj = slice_adjacency(A, node_ids, mask)
    del A
    t_adj = time.perf_counter() - t1
    t2 = time.perf_counter()
    per_slice, wl_info = wl_codes_batched(adj, max_iter=p["wl_max_iter"])
    t_wl = time.perf_counter() - t2
    codes = np.full(node_ids.shape, -1, np.int64)
    for i, c in enumerate(per_slice):
        codes[i, np.flatnonzero(mask[i])] = c
    n_edges = np.array([sum(len(nb) for nb in g) // 2 for g in adj])
    info = {
        "sampler": m["sampler"], "k": m["k"], "shape": list(codes.shape),
        "graph_rebuild_s": t_graph, "subgraph_extract_s": t_adj, "wl_s": t_wl,
        "total_build_s": time.perf_counter() - t0, "wl": wl_info,
        "edges_per_slice": {"min": int(n_edges.min()), "mean": float(n_edges.mean()),
                            "max": int(n_edges.max())},
        "stats_subgraph": code_stats(codes, mask),
        "stats_full": code_stats(data.wl_codes[np.where(mask, node_ids, 0)], mask),
    }
    np.savez(path, codes=codes, node_ids=node_ids, info=json.dumps(info))
    print(f"[{data.name}] subgraph WL: {wl_info['rounds']} rounds, {wl_info['n_codes']} codes, "
          f"{info['total_build_s']:.1f}s", flush=True)
    return {"codes": codes, "node_ids": node_ids, "info": info}
