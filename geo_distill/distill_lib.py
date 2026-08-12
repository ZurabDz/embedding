"""Shared helpers: tokenization, distillation losses, and evaluation metrics.

Imported by train.py and eval.py so the two stay perfectly consistent.
"""
from __future__ import annotations

import hashlib

import numpy as np
import jax
import jax.numpy as jnp
from tokenizers import Tokenizer


# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def encode_batch(tokenizer: Tokenizer, sentences, max_len: int, pad_id: int = 0):
    """Turn a list of strings into (tokens, pad_mask) numpy arrays.

    tokens:   (N, max_len) int32, right-padded with pad_id
    pad_mask: (N, max_len) float32, 1 for real tokens and 0 for padding
    """
    n = len(sentences)
    tokens = np.full((n, max_len), pad_id, dtype=np.int32)
    mask = np.zeros((n, max_len), dtype=np.float32)
    for i, enc in enumerate(tokenizer.encode_batch(list(sentences))):
        ids = enc.ids[:max_len]
        tokens[i, : len(ids)] = ids
        mask[i, : len(ids)] = 1.0
    return tokens, mask


# --------------------------------------------------------------------------- #
# Train / val split
# --------------------------------------------------------------------------- #
def val_split(sentences, val_frac: float, seed: int = 0):
    """Deterministic, content-based train/val split.

    A sentence lands in val or train based only on a hash of its *text* (plus the
    seed) — never on the dataset's size or order. So as you embed more data, every
    already-seen sentence keeps its assignment and the validation set stays stable,
    which makes metrics comparable across runs with different amounts of data.
    (The old random-permutation split re-drew a different val set every run, so the
    score moved as much because of the changing benchmark as the model.)

    Returns (train_idx, val_idx) as int arrays.
    """
    thr = int(val_frac * 1_000_000)
    train_idx, val_idx = [], []
    for i, s in enumerate(sentences):
        h = int(hashlib.sha1(f"{seed}\x00{s}".encode("utf-8")).hexdigest(), 16)
        (val_idx if (h % 1_000_000) < thr else train_idx).append(i)
    # Guard tiny corpora so neither split is ever empty.
    if len(val_idx) < 2 or not train_idx:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(sentences))
        n_val = max(2, int(len(sentences) * val_frac))
        val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    return np.array(train_idx), np.array(val_idx)


# --------------------------------------------------------------------------- #
# Distillation losses  (student embeddings vs teacher embeddings)
# cosine_regression_loss gives every example a full-information target (its own
# teacher vector): the student outputs directly in the teacher's embedding space
# and is pulled onto the teacher vector. The similarity losses only see the
# *pairwise cosine-similarity matrix* within each batch and work at any
# dimensionality.
# --------------------------------------------------------------------------- #
def _normalize(x, eps=1e-8):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


def cosine_regression_loss(student_emb, teacher_target):
    """Pull each student embedding onto its (unit-norm) teacher target:
    mean(1 - cos). Both inputs must have the same dimensionality."""
    s = _normalize(student_emb)
    t = _normalize(teacher_target)
    return jnp.mean(1.0 - jnp.sum(s * t, axis=-1))


def mse_sim_loss(student_emb, teacher_emb):
    """Match the two Gram (cosine-similarity) matrices with mean-squared error."""
    s = _normalize(student_emb)
    t = _normalize(teacher_emb)
    return jnp.mean((s @ s.T - t @ t.T) ** 2)


def kl_sim_loss(student_emb, teacher_emb, temperature: float = 0.05):
    """Distill a *soft retrieval distribution*.

    For every row we form a softmax over similarities to all other items in the
    batch (self excluded). The teacher's distribution is the target; we minimise
    cross-entropy against the student's. Lower temperature -> sharper targets,
    focusing on nearest neighbours. This tends to preserve ranking better than
    raw MSE when you mostly care about "which items are closest".
    """
    s = _normalize(student_emb)
    t = _normalize(teacher_emb)
    b = s.shape[0]
    neg_inf = jnp.finfo(s.dtype).min
    eye = jnp.eye(b, dtype=bool)

    s_logits = jnp.where(eye, neg_inf, (s @ s.T) / temperature)
    t_logits = jnp.where(eye, neg_inf, (t @ t.T) / temperature)

    p = jax.nn.softmax(t_logits, axis=-1)
    logq = jax.nn.log_softmax(s_logits, axis=-1)
    return -jnp.sum(p * logq, axis=-1).mean()


# --------------------------------------------------------------------------- #
# Metrics (numpy — run outside the jitted training step)
# --------------------------------------------------------------------------- #
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


def similarity_agreement(student_emb, teacher_emb, max_pairs: int = 4_000_000):
    """How well does the student reproduce the teacher's similarity geometry?

    Returns Pearson & Spearman correlation between the two off-diagonal
    similarity matrices, plus top-1 nearest-neighbour agreement.
    """
    s = np.asarray(student_emb, dtype=np.float64)
    t = np.asarray(teacher_emb, dtype=np.float64)
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
