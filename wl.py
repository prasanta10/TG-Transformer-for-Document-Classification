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
