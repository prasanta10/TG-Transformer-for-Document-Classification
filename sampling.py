"""Context samplers: build the per-document input slice [target, k helper words].

All samplers return global node ids of shape (n_docs, k+1) plus a boolean mask
(True = real node, False = padding). Column 0 is always the target document.
Padding positions point at ``pad_id`` (an extra all-zero row appended to the node
embedding table) and are masked in attention and readout.
"""

from __future__ import annotations

import numpy as np


def intimacy_helpers(helpers_top: np.ndarray, k: int) -> np.ndarray:
    """Top-k word nodes by intimacy (paper Sec 2.2): the first k of the cached top-50.

    Returns:
        (n_docs, k) int64 global node ids (never padded).
    """
    if k > helpers_top.shape[1]:
        raise ValueError(f"k={k} exceeds cached helper list length {helpers_top.shape[1]}")
    return helpers_top[:, :k]


def tfidf_helpers(tfidf, n_docs: int, k: int, pad_id: int) -> np.ndarray:
    """The document's own words ranked by TF-IDF weight (ties: smaller word id first).

    Only words with positive TF-IDF weight are candidates (words occurring in every
    document have IDF 0 and no graph edge). Documents with fewer than k such words
    are padded with ``pad_id``.

    Returns:
        (n_docs, k) int64 global node ids, ``pad_id`` where padded.
    """
    out = np.full((tfidf.shape[0], k), pad_id, np.int64)
    for i in range(tfidf.shape[0]):
        lo, hi = tfidf.indptr[i], tfidf.indptr[i + 1]
        cols, vals = tfidf.indices[lo:hi], tfidf.data[lo:hi]
        order = np.lexsort((cols, -vals))[:k]
        out[i, :len(order)] = cols[order] + n_docs
    return out


def random_helpers(n_docs: int, n_words: int, k: int, seed: int) -> np.ndarray:
    """k distinct random vocabulary words per document, drawn with ``RandomState(seed)``.

    Returns:
        (n_docs, k) int64 global node ids.
    """
    rng = np.random.RandomState(seed)
    out = np.empty((n_docs, k), np.int64)
    for i in range(n_docs):
        out[i] = rng.choice(n_words, size=k, replace=False) + n_docs
    return out


def build_slices(data, sampler: str, k: int, seed: int = 0):
    """Build input slices [target, k helpers] for every document.

    Args:
        data: ``data.PreparedData``.
        sampler: one of ``intimacy`` | ``tfidf`` | ``random``.
        k: number of helper words.
        seed: seed for the ``random`` sampler.

    Returns:
        node_ids: (n_docs, k+1) int64; column 0 is the document itself.
        mask: (n_docs, k+1) bool; True for real nodes.
        pad_id: the padding node id (== data.n_nodes).
    """
    pad_id = data.n_nodes
    if sampler == "intimacy":
        h = intimacy_helpers(data.helpers, k)
    elif sampler == "tfidf":
        h = tfidf_helpers(data.tfidf, data.n_docs, k, pad_id)
    elif sampler == "random":
        h = random_helpers(data.n_docs, data.n_words, k, seed)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'")
    node_ids = np.concatenate([np.arange(data.n_docs, dtype=np.int64)[:, None], h], axis=1)
    mask = node_ids != pad_id
    return node_ids, mask, pad_id


def resample_intimacy_helpers(node_ids: np.ndarray, helpers_top: np.ndarray,
                              helper_scores: np.ndarray, rows: np.ndarray, k: int,
                              seed: int, epoch: int) -> np.ndarray:
    """Redraw the helpers of the documents in ``rows`` (R-all ``helper_resample``).

    For each document in ``rows``, k distinct helpers are drawn without replacement
    from its cached top-50 intimacy list, with probability proportional to intimacy.
    The draw uses the Gumbel-top-k trick (the k largest log(score) + Gumbel noise),
    which has exactly the distribution of drawing one helper at a time without
    replacement. Random numbers come from ``default_rng([seed, epoch])``, separate
    from torch's generators. Drawn helpers are stored in descending intimacy order.
    All other rows (validation / test documents) are returned unchanged.

    Args:
        node_ids: (n_docs, k+1) fixed slices; column 0 is the document.
        helpers_top / helper_scores: (n_docs, 50) cached helper ids and intimacy scores.
        rows: document indices to resample (the training documents).

    Returns:
        (n_docs, k+1) int64 copy of ``node_ids`` with the given rows resampled.
    """
    if np.any(helper_scores[rows] <= 0):
        raise ValueError("intimacy scores must be positive for proportional sampling")
    rng = np.random.default_rng([seed, epoch])
    keys = np.log(helper_scores[rows].astype(np.float64)) + rng.gumbel(size=(len(rows), helpers_top.shape[1]))
    pick = np.argpartition(-keys, k - 1, axis=1)[:, :k]
    pick.sort(axis=1)                       # cached lists are in descending intimacy order
    out = node_ids.copy()
    out[rows, 1:] = np.take_along_axis(helpers_top[rows], pick, axis=1)
    return out
