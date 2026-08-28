"""Distillation losses (student embeddings vs teacher embeddings).

cosine_regression_loss gives every example a full-information target (its own
teacher vector): the student outputs directly in the teacher's embedding space
and is pulled onto the teacher vector. The similarity losses only see the
*pairwise cosine-similarity matrix* within each batch and work at any
dimensionality.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from geo_distill.model import l2_normalize


def cosine_regression_loss(student_emb, teacher_target):
    """Pull each student embedding onto its (unit-norm) teacher target:
    mean(1 - cos). Both inputs must have the same dimensionality."""
    s = l2_normalize(student_emb)
    t = l2_normalize(teacher_target)
    return jnp.mean(1.0 - jnp.sum(s * t, axis=-1))


def mse_sim_loss(student_emb, teacher_emb):
    """Match the two Gram (cosine-similarity) matrices with mean-squared error."""
    s = l2_normalize(student_emb)
    t = l2_normalize(teacher_emb)
    return jnp.mean((s @ s.T - t @ t.T) ** 2)


def kl_sim_loss(student_emb, teacher_emb, temperature: float = 0.05):
    """Distill a *soft retrieval distribution*.

    For every row we form a softmax over similarities to all other items in the
    batch (self excluded). The teacher's distribution is the target; we minimise
    cross-entropy against the student's. Lower temperature -> sharper targets,
    focusing on nearest neighbours. This tends to preserve ranking better than
    raw MSE when you mostly care about "which items are closest".
    """
    s = l2_normalize(student_emb)
    t = l2_normalize(teacher_emb)
    b = s.shape[0]
    neg_inf = jnp.finfo(s.dtype).min
    eye = jnp.eye(b, dtype=bool)

    s_logits = jnp.where(eye, neg_inf, (s @ s.T) / temperature)
    t_logits = jnp.where(eye, neg_inf, (t @ t.T) / temperature)

    p = jax.nn.softmax(t_logits, axis=-1)
    logq = jax.nn.log_softmax(s_logits, axis=-1)
    return -jnp.sum(p * logq, axis=-1).mean()
