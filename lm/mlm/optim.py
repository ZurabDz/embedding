"""Learning-rate schedule and optimizer."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


def trapezoid_schedule(peak: float, total: int, warmup_frac=0.03, decay_frac=0.15):
    """Warmup -> flat -> 1-sqrt decay.

    The flat plateau is what makes the run length a free parameter: you can stop
    anywhere, or fork a decay branch from any checkpoint on the plateau.
    """
    warmup = max(1, int(total * warmup_frac))
    decay = max(1, int(total * decay_frac))
    stable = max(1, total - warmup - decay)

    def one_minus_sqrt(count):
        p = jnp.clip(count / decay, 0.0, 1.0)
        return peak * (1.0 - jnp.sqrt(p))

    return optax.join_schedules(
        [optax.linear_schedule(0.0, peak, warmup),
         optax.constant_schedule(peak),
         one_minus_sqrt],
        boundaries=[warmup, warmup + stable],
    )


def make_optimizer(model, schedule, weight_decay=0.1, clip=1.0):
    # Decay only the 2-D weight matrices. The 1-D RMSNorm scales are the
    # per-channel gains the whole residual stream passes through; pulling them
    # toward zero shrinks activations globally rather than regularising.
    no_1d = lambda tree: jax.tree.map(lambda x: jnp.ndim(x) > 1, tree)
    tx = optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adamw(schedule, b1=0.9, b2=0.95, eps=1e-8,
                    weight_decay=weight_decay, mask=no_1d),
    )
    return nnx.Optimizer(model, tx, wrt=nnx.Param)
