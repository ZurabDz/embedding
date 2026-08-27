"""The bidirectional MLM encoder, its sublayers, and its losses.

Architecture follows the ModernBERT/NeoBERT consensus: RoPE, pre-RMSNorm,
SwiGLU, no biases except the tied decoder's, tied embeddings.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from mlm.config import EncoderConfig

# ----------------------------------------------------------------------------
# Rotary position embedding
# ----------------------------------------------------------------------------


def rope_tables(seq_len: int, head_dim: int, theta: float, dtype):
    """cos/sin lookup tables of shape [seq_len, head_dim].

    Recomputed on every forward pass on purpose: seq_len is static under jit,
    so XLA constant-folds this away. Keeps the module free of buffers.
    """
    inv_freq = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    pos = jnp.arange(seq_len, dtype=jnp.float32)
    ang = pos[:, None] * inv_freq[None, :]  # [L, head_dim/2]
    cos = jnp.concatenate([jnp.cos(ang), jnp.cos(ang)], axis=-1)
    sin = jnp.concatenate([jnp.sin(ang), jnp.sin(ang)], axis=-1)
    return cos.astype(dtype), sin.astype(dtype)


def _rotate_half(x):
    a, b = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-b, a], axis=-1)


def apply_rope(x, cos, sin):
    """x: [B, H, L, D]; cos/sin: [L, D]. Applied to queries and keys only."""
    return x * cos[None, None] + _rotate_half(x) * sin[None, None]


def attention_bias(attention_mask, dtype):
    """[B, L] padding mask -> additive [B, 1, 1, L] bias.

    Keys are masked only where padded, so every real position attends to every
    other real position in both directions.
    """
    # halved so it stays representable in every float dtype: a bare -1e9 becomes
    # -inf under fp16, and -inf - (-inf) in the softmax is NaN
    neg = jnp.asarray(jnp.finfo(dtype).min / 2, dtype)
    keep = attention_mask[:, None, None, :].astype(bool)  # [B, 1, 1, L]
    return jnp.where(keep, jnp.zeros((), dtype), neg)


# ----------------------------------------------------------------------------
# Sublayers
# ----------------------------------------------------------------------------


@functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)
def _attend(q, k, v, bias, scale):
    """Scores -> softmax -> values, rematerialised in the backward pass.

    The [B,H,L,L] score matrix is the single largest activation in the model —
    192 MiB per layer at batch 32, and one copy survives per layer as the
    softmax's residual, so 2.25 GiB across 12 layers. Recomputing it costs about
    4% more FLOPs and is the best memory-per-FLOP trade available here.

    Deliberately not jax.nn.dot_product_attention: its 'xla' path performs this
    exact sequence and materialises exactly as much, and its fused 'cudnn' path
    rejects float32 outright and needs Ampere, so it is unavailable on the T4
    this has to run on.
    """
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    # softmax in fp32 regardless of compute dtype
    w = jax.nn.softmax((scores + bias).astype(jnp.float32), axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bhkd->bhqd", w, v)


class BidirectionalAttention(nnx.Module):
    def __init__(self, cfg: EncoderConfig, *, out_std: float, rngs: nnx.Rngs):
        self.cfg = cfg
        kin = nnx.initializers.normal(cfg.init_std)
        kout = nnx.initializers.normal(out_std)
        lin = lambda i, o, init: nnx.Linear(
            i, o, use_bias=False, kernel_init=init, dtype=cfg.dtype,
            param_dtype=cfg.param_dtype, rngs=rngs,
        )
        self.wq = lin(cfg.hidden, cfg.hidden, kin)
        self.wk = lin(cfg.hidden, cfg.hidden, kin)
        self.wv = lin(cfg.hidden, cfg.hidden, kin)
        self.wo = lin(cfg.hidden, cfg.hidden, kout)
        self.drop = nnx.Dropout(cfg.dropout, rngs=rngs)

    def __call__(self, x, bias, cos, sin):
        cfg = self.cfg
        b, l, _ = x.shape
        h, d = cfg.heads, cfg.head_dim

        split = lambda t: t.reshape(b, l, h, d).transpose(0, 2, 1, 3)  # [B,H,L,D]
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cfg.dropout > 0.0:
            # Dropout belongs on the attention weights, and the module's RNG
            # cannot cross into a pure checkpointed function — so when it is on,
            # take the unrematerialised path rather than quietly moving where the
            # noise is applied. cfg.dropout is static, so this costs no runtime
            # branch. ModernBERT trains at 0.0, which is the default here.
            scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(d)
            weights = jax.nn.softmax((scores + bias).astype(jnp.float32), axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", self.drop(weights.astype(x.dtype)), v)
        else:
            out = _attend(q, k, v, bias, 1.0 / math.sqrt(d))
        out = out.transpose(0, 2, 1, 3).reshape(b, l, cfg.hidden)
        return self.wo(out)


class SwiGLU(nnx.Module):
    def __init__(self, cfg: EncoderConfig, *, out_std: float, rngs: nnx.Rngs):
        kin = nnx.initializers.normal(cfg.init_std)
        kout = nnx.initializers.normal(out_std)
        lin = lambda i, o, init: nnx.Linear(
            i, o, use_bias=False, kernel_init=init, dtype=cfg.dtype,
            param_dtype=cfg.param_dtype, rngs=rngs,
        )
        self.gate = lin(cfg.hidden, cfg.mlp_hidden, kin)
        self.up = lin(cfg.hidden, cfg.mlp_hidden, kin)
        self.down = lin(cfg.mlp_hidden, cfg.hidden, kout)

    def __call__(self, x):
        return self.down(jax.nn.silu(self.gate(x)) * self.up(x))


class EncoderBlock(nnx.Module):
    def __init__(self, cfg: EncoderConfig, *, out_std: float, rngs: nnx.Rngs):
        norm = lambda: nnx.RMSNorm(
            cfg.hidden, epsilon=cfg.norm_eps, dtype=cfg.dtype,
            param_dtype=cfg.param_dtype, rngs=rngs,
        )
        self.attn_norm = norm()
        self.attn = BidirectionalAttention(cfg, out_std=out_std, rngs=rngs)
        self.mlp_norm = norm()
        self.mlp = SwiGLU(cfg, out_std=out_std, rngs=rngs)
        self.drop = nnx.Dropout(cfg.dropout, rngs=rngs)

    def __call__(self, x, bias, cos, sin):
        x = x + self.drop(self.attn(self.attn_norm(x), bias, cos, sin))
        x = x + self.drop(self.mlp(self.mlp_norm(x)))
        return x


class TiedEmbedding(nnx.Module):
    """One matrix used both as the input lookup and the output projection."""

    def __init__(self, cfg: EncoderConfig, *, rngs: nnx.Rngs):
        self.embed = nnx.Embed(
            cfg.vocab_size, cfg.hidden,
            embedding_init=nnx.initializers.normal(cfg.init_std),
            dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs,
        )
        # Per-vocab output bias, as in BERT and ModernBERT. Without it the tied
        # table has to serve two masters — a small input lookup and an output
        # projection that wants to encode the unigram prior. Starts at zero so
        # the init loss is still exactly ln(vocab). 1-D, so the optimizer's
        # ndim > 1 rule excludes it from weight decay for free.
        self.bias = nnx.Param(jnp.zeros((cfg.vocab_size,), cfg.param_dtype))

    def encode(self, ids):
        return self.embed(ids)

    def decode(self, x):
        # cast the table down to the compute dtype rather than letting the
        # matmul promote x up: this is the widest matmul in the model
        w = self.embed.embedding[...].astype(x.dtype)
        return x @ w.T + self.bias[...].astype(x.dtype)


class MLMHead(nnx.Module):
    """Dense -> GELU -> norm. The vocab projection lives in TiedEmbedding."""

    def __init__(self, cfg: EncoderConfig, *, rngs: nnx.Rngs):
        self.dense = nnx.Linear(
            cfg.hidden, cfg.hidden, use_bias=False,
            kernel_init=nnx.initializers.normal(cfg.init_std),
            dtype=cfg.dtype, param_dtype=cfg.param_dtype, rngs=rngs,
        )
        self.norm = nnx.RMSNorm(
            cfg.hidden, epsilon=cfg.norm_eps, dtype=cfg.dtype,
            param_dtype=cfg.param_dtype, rngs=rngs,
        )

    def __call__(self, x):
        return self.norm(jax.nn.gelu(self.dense(x)))


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------


@functools.partial(nnx.remat,
                   policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
def _remat_block(block, x, bias, cos, sin):
    """One encoder block with its activations recomputed in the backward pass.

    The policy saves the Linear matmuls (no batch dims) and recomputes the
    attention einsums (which carry b,h batch dims) — the right split, at ~4%
    more FLOPs. nnx.remat rejects bound methods, so `block` has to arrive as an
    ordinary first argument rather than as `block.__call__`.
    """
    return block(x, bias, cos, sin)


class MlmEncoder(nnx.Module):
    def __init__(self, cfg: EncoderConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        # scale residual-path output projections by 1/sqrt(2 * n_layers)
        out_std = cfg.init_std / math.sqrt(2 * cfg.layers)
        self.tok = TiedEmbedding(cfg, rngs=rngs)
        self.embed_drop = nnx.Dropout(cfg.dropout, rngs=rngs)
        # flax >= 0.12 requires containers of modules to be declared as data
        self.blocks = nnx.List(
            [EncoderBlock(cfg, out_std=out_std, rngs=rngs) for _ in range(cfg.layers)]
        )
        self.final_norm = nnx.RMSNorm(
            cfg.hidden, epsilon=cfg.norm_eps, dtype=cfg.dtype,
            param_dtype=cfg.param_dtype, rngs=rngs,
        )
        self.head = MLMHead(cfg, rngs=rngs)

    def _run_blocks(self, x, bias, cos, sin):
        if self.cfg.remat:
            for block in self.blocks:
                x = _remat_block(block, x, bias, cos, sin)
        else:
            for block in self.blocks:
                x = block(x, bias, cos, sin)
        return x

    def encode(self, input_ids, attention_mask):
        """Returns hidden states [B, L, hidden] — use this for embeddings."""
        cfg = self.cfg
        _, l = input_ids.shape
        cos, sin = rope_tables(l, cfg.head_dim, cfg.rope_theta, cfg.dtype)
        bias = attention_bias(attention_mask, cfg.dtype)

        x = self.embed_drop(self.tok.encode(input_ids))
        x = self._run_blocks(x, bias, cos, sin)
        return self.final_norm(x)

    def __call__(self, input_ids, attention_mask, positions=None):
        """Logits [B, L, vocab] — or [B, K, vocab] when `positions` is given.

        Gathering the masked positions before the head is the difference between
        projecting 512 positions into a 32k vocabulary and projecting the ~154
        that are actually scored. K is a Python int fixed for the run, so the
        gather is statically shaped and nothing retraces.
        """
        x = self.encode(input_ids, attention_mask)
        if positions is not None:
            x = jnp.take_along_axis(x, positions[:, :, None], axis=1)  # [B, K, hidden]
        return self.tok.decode(self.head(x))


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------


def _masked_ce(logits, targets, weights):
    """Cross-entropy over the gathered targets, averaged by weight.

    The fp32 cast is load-bearing under --dtype bfloat16: optax's
    softmax_cross_entropy_with_integer_labels goes through jax.nn.logsumexp,
    which does not promote, so the sum over 32k vocabulary entries would be
    accumulated with 8 mantissa bits.
    """
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets)
    return (ce * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def mlm_loss(model: MlmEncoder, batch):
    logits = model(batch["input_ids"], batch["attention_mask"], batch["positions"])
    return _masked_ce(logits, batch["targets"], batch["weights"])


def count_params(model) -> dict[str, int]:
    state = nnx.state(model, nnx.Param)
    flat = nnx.to_flat_state(state)
    total, embedding = 0, 0
    for path, var in flat:
        size = int(np.prod(var[...].shape))
        total += size
        if "embedding" in path:
            embedding += size
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}
