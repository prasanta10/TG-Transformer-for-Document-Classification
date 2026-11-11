"""Weisfeiler-Lehman node codes (paper Sec 2.3, "Weisfeiler-Lehman Structural Encoding").

Following Graph-BERT (Zhang et al., 2020a), WL color refinement is run once on the
full graph and yields one integer code per node, later turned into a sinusoidal
encoding (paper Eq. 5, see ``model.wl_sinusoidal_encoding``).
"""

from __future__ import annotations

import numpy as np


def wl_codes(A, max_iter: int = 10):
    """Weisfeiler-Lehman color refinement on the unweighted graph.

    The graph is the structure of ``A`` with edge weights ignored and self-loops
    removed. All nodes start with the same color 0 (Graph-BERT convention). At each
    iteration a node's new color is an injective relabelling of the signature
    (own color, sorted multiset of neighbour colors). New integer ids are assigned
    in sorted order of the signatures (Graph-BERT also relabels via sorted
    signatures), which makes the codes deterministic and invariant to node order.
    Refinement only ever splits color classes, so it has stabilised as soon as the
    number of colors stops increasing. Iteration is capped at ``max_iter``.

    Args:
        A: (n, n) sparse adjacency (weights and diagonal are ignored).
        max_iter: iteration cap.

    Returns:
        codes: (n,) int64 WL code per node (stable colors).
        info: dict with iterations, convergence flag and colors per iteration.
    """
    A = A.tocsr().copy()
    A.setdiag(0)
    A.eliminate_zeros()
    indptr, indices = A.indptr, A.indices
    n = A.shape[0]
    colors = np.zeros(n, np.int64)
    history = [1]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        sigs = [(int(colors[v]), np.sort(colors[indices[indptr[v]:indptr[v + 1]]]).tobytes())
                for v in range(n)]
        # Canonical relabelling: new ids follow the sorted order of the signatures,
        # so codes depend only on graph structure, never on node order (e.g. on a
        # corpus file sorted by class). Numbering by first appearance would make the
        # near-unique codes a proxy for the node index and leak labels.
        table = {s: i for i, s in enumerate(sorted(set(sigs)))}
        colors = np.fromiter((table[s] for s in sigs), dtype=np.int64, count=n)
        n_colors = len(table)
        history.append(n_colors)
        if n_colors == history[-2]:
            converged = True
            break
    info = {"iterations": it, "converged": converged, "n_colors": int(history[-1]),
            "colors_per_iter": history, "max_code": int(colors.max())}
    return colors, info


def wl_codes_batched(adj_lists, max_iter: int = 10):
    """WL colour refinement on many small graphs at once, with one shared dictionary.

    Used for W1 (WL computed inside each sampled slice). All graphs are refined in
    lockstep: every node starts with colour 0, and in each round its signature is
    (own colour, sorted multiset of neighbour colours). One global dictionary per
    round maps signatures to new integer ids in sorted signature order, so ids are
    deterministic, independent of node order, and comparable across graphs. A graph
    whose partition has stopped changing keeps the same partition in later rounds
    (refinement only splits classes), so iterating until EVERY graph is stable (or
    ``max_iter`` rounds) gives each graph its own stable partition while all codes
    come from the same round.

    Args:
        adj_lists: list of graphs; each graph is a list of neighbour-index lists
            (unweighted, no self-loops), one per node.
        max_iter: round cap.

    Returns:
        codes: list of int64 arrays, one per graph (codes start at 0).
        info: dict with rounds, convergence flag and number of distinct codes per round.
    """
    colors = [np.zeros(len(g), np.int64) for g in adj_lists]
    n_parts = [1 if len(g) else 0 for g in adj_lists]
    history = [1]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        sigs = [[(int(c[v]), tuple(sorted(int(c[u]) for u in nb))) for v, nb in enumerate(g)]
                for g, c in zip(adj_lists, colors)]
        table = {s: i for i, s in enumerate(sorted({s for gs in sigs for s in gs}))}
        colors = [np.fromiter((table[s] for s in gs), np.int64, count=len(gs)) for gs in sigs]
        new_parts = [len(set(c.tolist())) for c in colors]
        history.append(len(table))
        stable = new_parts == n_parts
        n_parts = new_parts
        if stable:
            converged = True
            break
    info = {"rounds": it, "converged": converged, "codes_per_round": history,
            "n_codes": history[-1]}
    return colors, info
