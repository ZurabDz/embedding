"""Evaluation metrics (numpy — run outside the jitted training step)."""
from __future__ import annotations

import numpy as np

from geo_distill.config import DEFAULT_METRIC_ROWS


def _upper_tri(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12
    return float((a * b).sum() / denom)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return _pearson(ra.astype(np.float64), rb.astype(np.float64))


# Two dense n x n float64 matrices are 16*n^2 bytes, so the metric's cost is
# quadratic in the row count and nothing else: the default 4096 rows cost
# ~270 MB and give 8.4M pairs (Spearman is long since converged), while 50k
# rows would ask for 40 GB.
# Refuse rather than let the kernel decide. An allocation this size is not a
# MemoryError a caller can catch — it is an OOM kill that takes the whole
# training run with it, after the epoch and before any checkpoint.
MAX_METRIC_ROWS = 10_000


def similarity_agreement(student_emb, teacher_emb, max_pairs: int = 4_000_000):
    """How well does the student reproduce the teacher's similarity geometry?

    Returns Pearson & Spearman correlation between the two off-diagonal
    similarity matrices, plus top-1 nearest-neighbour agreement.

    Quadratic in host memory — score a subset (see data.metric_subset), not a
    whole large val split.
    """
    n_rows = len(student_emb)
    if n_rows > MAX_METRIC_ROWS:
        raise ValueError(
            f"similarity_agreement got {n_rows} rows: the two dense "
            f"{n_rows}x{n_rows} similarity matrices alone need "
            f"{16 * n_rows ** 2 / 1e9:.0f} GB of host RAM, which the kernel "
            f"answers with an OOM kill rather than an exception. Score a "
            f"subsample instead — `train` and `eval` take --val-metric-n "
            f"(default {DEFAULT_METRIC_ROWS}) for exactly this.")
    # np.array (not asarray): a float64 input would otherwise alias the
    # caller's array and the in-place normalization below would mutate it.
    s = np.array(student_emb, dtype=np.float64)
    t = np.array(teacher_emb, dtype=np.float64)
    s /= (np.linalg.norm(s, axis=-1, keepdims=True) + 1e-12)
    t /= (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-12)

    s_sim = s @ s.T
    t_sim = t @ t.T

    su = _upper_tri(s_sim)
    tu = _upper_tri(t_sim)
    if su.size > max_pairs:  # subsample for very large val sets
        idx = np.random.default_rng(0).choice(su.size, max_pairs, replace=False)
        su, tu = su[idx], tu[idx]

    # top-1 nearest neighbour, excluding self
    n = s_sim.shape[0]
    np.fill_diagonal(s_sim, -np.inf)
    np.fill_diagonal(t_sim, -np.inf)
    s_nn = s_sim.argmax(axis=1)
    t_nn = t_sim.argmax(axis=1)
    nn_agree = float((s_nn == t_nn).mean()) if n > 1 else float("nan")

    return {
        "pearson": _pearson(su, tu),
        "spearman": _spearman(su, tu),
        "top1_nn_agreement": nn_agree,
    }
