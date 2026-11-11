"""Heterogeneous text graph (paper Sec 2.1) and intimacy matrix (paper Eq. 2).

Node order: all documents (train and test) first, then all vocabulary words, so
global node id of word ``j`` is ``n_docs + j``.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp


def doc_word_tfidf(docs_idx, n_words):
    """Document-word TF-IDF matrix (TextGCN convention).

    TF = raw count of the word in the document; IDF = log(N_docs / df). Words that
    occur in every document get IDF 0 and therefore no edge (zeros are removed).

    Returns:
        (n_docs, n_words) CSR float64 matrix.
    """
    n_docs = len(docs_idx)
    rows = np.concatenate([np.full(len(d), i, np.int64) for i, d in enumerate(docs_idx)])
    cols = np.concatenate(docs_idx)
    counts = sp.csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(n_docs, n_words))
    counts.sum_duplicates()
    df = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log(n_docs / np.maximum(df, 1))
    tfidf = counts.multiply(idf[None, :]).tocsr()
    tfidf.eliminate_zeros()
    return tfidf


def _window_incidence(docs_idx, n_words, window: int):
    """Binary window-by-word incidence matrix W for a list of documents.

    TextGCN convention: a document of length <= ``window`` forms a single window;
    longer documents yield ``len - window + 1`` sliding windows. Windows never
    cross document boundaries. ``W[w, j] = 1`` iff word j occurs in window w.
    """
    rows, cols = [], []
    n_win = 0
    for ids in docs_idx:
        L = len(ids)
        if L == 0:
            continue
        if L <= window:
            rows.append(np.full(L, n_win, np.int64))
            cols.append(ids)
            n_win += 1
        else:
            nw = L - window + 1
            idx = np.arange(nw)[:, None] + np.arange(window)[None, :]
            rows.append(np.repeat(np.arange(n_win, n_win + nw), window))
            cols.append(ids[idx].ravel())
            n_win += nw
    if n_win == 0:
        return sp.csr_matrix((0, n_words))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    W = sp.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(n_win, n_words))
    W.sum_duplicates()
    W.data[:] = 1.0  # a window "contains" a word regardless of multiplicity
    return W


def word_word_pmi(docs_idx, n_words, window: int = 20, chunk_docs: int = 500):
    """Positive PMI between words over sliding windows. Implements paper Eq. 1.

    PMI(i, j) = log( N_ij * N / (N_i * N_j) ), where N_i is the number of windows
    containing word i, N_ij the number containing both i and j, and N the total
    number of windows. Co-occurrence counts are obtained as C = W^T W from the
    binary window-word incidence matrix (accumulated in document chunks).
    Only pairs i != j with PMI > 0 are kept.

    Returns:
        P: (n_words, n_words) symmetric CSR matrix of positive PMI weights.
        n_windows: total number of windows N.
    """
    C = sp.csr_matrix((n_words, n_words))
    n_windows = 0
    for s in range(0, len(docs_idx), chunk_docs):
        W = _window_incidence(docs_idx[s:s + chunk_docs], n_words, window)
        n_windows += W.shape[0]
        C = C + (W.T @ W).tocsr()
    Ni = C.diagonal()
    C = C.tocoo()
    off = C.row != C.col
    r, c, nij = C.row[off], C.col[off], C.data[off]
    pmi = np.log(nij * n_windows / (Ni[r] * Ni[c]))
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (r[keep], c[keep])), shape=(n_words, n_words))
    return P, n_windows


def build_text_graph(docs_idx, n_words, window: int = 20):
    """Build the symmetric weighted adjacency A + I of the text graph (paper Sec 2.1).

    Blocks: doc-doc = 0; doc-word = TF-IDF; word-word = positive PMI (Eq. 1).
    Self-loops (identity) are added to every node.

    Returns:
        A: (n, n) CSR float64 adjacency including self-loops.
        tfidf: (n_docs, n_words) CSR document-word TF-IDF matrix.
        stats: dict with edge counts.
    """
    n_docs = len(docs_idx)
    n = n_docs + n_words
    tfidf = doc_word_tfidf(docs_idx, n_words)
    P, n_windows = word_word_pmi(docs_idx, n_words, window)
    A = sp.bmat([[sp.csr_matrix((n_docs, n_docs)), tfidf], [tfidf.T, P]], format="csr")
    A = (A + sp.identity(n, format="csr")).tocsr()
    A.sum_duplicates()
    sym_err = abs(A - A.T).max()
    stats = {
        "n_nodes": int(n), "n_docs": int(n_docs), "n_words": int(n_words),
        "doc_word_edges": int(tfidf.nnz), "word_word_edges": int(P.nnz // 2),
        "n_windows": int(n_windows), "nnz_A_with_self_loops": int(A.nnz),
        "symmetry_max_abs_err": float(sym_err),
        "isolated_nodes_excl_self_loop": int(np.sum(np.diff(A.indptr) == 1)),
    }
    return A, tfidf, stats


def normalized_adjacency(A):
    """Abar = D^-1/2 A D^-1/2 with D the weighted degree of A (self-loops included)."""
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(deg)
    Dm = sp.diags(dinv)
    return (Dm @ A @ Dm).tocsr()


def intimacy_and_helpers(A, n_docs: int, alpha: float = 0.15, top_n: int = 50,
                         block_size: int = 2048):
    """Compute the intimacy matrix S and each document's top word helpers.

    Implements paper Eq. 2: S = alpha * (I - (1 - alpha) * Abar)^-1.

    M = I - (1 - alpha) * Abar is symmetric positive definite (eigenvalues in
    [alpha, 2 - alpha]), so we Cholesky-factor the dense float32 M once (in place)
    and obtain S block-of-columns by block-of-columns via triangular solves. Since
    S is symmetric, column block = row block. This is the exact dense inverse, but
    the full n x n S is never held in memory at once (peak ~ one n x n float32
    matrix plus two n x block_size blocks).

    For every row we record min/max/sum and whether the diagonal is the largest
    entry. For each DOCUMENT row we keep the ``top_n`` WORD nodes by intimacy
    (documents, including the node itself, are excluded), sorted descending.

    Returns:
        dict with ``helpers`` (n_docs, top_n) int64 global node ids,
        ``helper_scores`` (n_docs, top_n) float32, ``stats`` and timings.
    """
    n = A.shape[0]
    n_words = n - n_docs
    t0 = time.perf_counter()
    Abar = normalized_adjacency(A).astype(np.float32)
    M = Abar.toarray()
    del Abar
    M *= -(1.0 - alpha)
    M[np.diag_indices(n)] += 1.0
    # M is symmetric, so M.T (a Fortran-ordered view of the same buffer) is the same
    # matrix; passing it lets LAPACK factor truly in place instead of copying.
    cf = sla.cho_factor(M.T, lower=True, overwrite_a=True, check_finite=False)
    t_solve = time.perf_counter() - t0
    t_help = 0.0

    row_min = np.empty(n, np.float32)
    row_max = np.empty(n, np.float32)
    row_sum = np.empty(n, np.float64)
    diag = np.empty(n, np.float32)
    other_max = np.empty(n, np.float32)
    helpers = np.empty((n_docs, top_n), np.int64)
    scores = np.empty((n_docs, top_n), np.float32)

    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        t1 = time.perf_counter()
        E = np.zeros((n, e - s), np.float32)
        E[np.arange(s, e), np.arange(e - s)] = 1.0
        Sb = sla.cho_solve(cf, E, overwrite_b=True, check_finite=False)
        Sb *= alpha
        t_solve += time.perf_counter() - t1

        # row statistics (column j of the block == row s+j of S)
        cols = np.arange(e - s)
        row_min[s:e] = Sb.min(axis=0)
        row_max[s:e] = Sb.max(axis=0)
        row_sum[s:e] = Sb.sum(axis=0, dtype=np.float64)
        diag[s:e] = Sb[np.arange(s, e), cols]
        Sb[np.arange(s, e), cols] = -np.inf
        other_max[s:e] = Sb.max(axis=0)
        Sb[np.arange(s, e), cols] = diag[s:e]

        # helper lists for document rows in this block
        if s < n_docs:
            t2 = time.perf_counter()
            d_end = min(e, n_docs)
            W = Sb[n_docs:, : d_end - s].T          # (docs in block, n_words)
            part = np.argpartition(-W, top_n - 1, axis=1)[:, :top_n]
            vals = np.take_along_axis(W, part, axis=1)
            # sort by descending score, ties broken by smaller word id
            order = np.lexsort((part, -vals), axis=1)
            part = np.take_along_axis(part, order, axis=1)
            helpers[s:d_end] = part + n_docs
            scores[s:d_end] = np.take_along_axis(vals, order, axis=1)
            t_help += time.perf_counter() - t2
        del Sb
    del cf, M

    def summ(v):
        """min/mean/max summary of a per-row statistic."""
        return {"min": float(v.min()), "mean": float(v.mean()), "max": float(v.max())}

    stats = {
        "n_rows": int(n), "alpha": alpha,
        "row_min": summ(row_min), "row_max": summ(row_max), "row_sum": summ(row_sum),
        "row_sum_docs": summ(row_sum[:n_docs]), "row_sum_words": summ(row_sum[n_docs:]),
        "diagonal": summ(diag),
        "diag_is_row_max": int(np.sum(diag >= row_max)),
        "diag_strictly_largest": int(np.sum(diag > other_max)),
        "negative_entries_rows": int(np.sum(row_min < 0)),
        "n_words": int(n_words),
    }
    return {"helpers": helpers, "helper_scores": scores, "stats": stats,
            "time_S_s": t_solve, "time_helpers_s": t_help}
