#!/usr/bin/env python3
"""
A small bidirectional MLM encoder in Flax NNX — model, masking, and training loop
in one file. Architecture follows the ModernBERT/NeoBERT consensus: RoPE,
pre-RMSNorm, SwiGLU, no biases except the tied decoder's, tied embeddings, 30%
masking with 100% [MASK] replacement.

Quick start, no network needed:
    python lm.py --selftest
    python lm.py --smoke

Real run on Georgian FineWeb-2:
    pip install datasets tokenizers
    python lm.py --steps 20000 --docs 200000

Corpora too large to hold in host RAM are tokenised once, up front, and then
streamed off disk — which also keeps a GPU session from spending its time on
CPU work:
    python lm.py --tokenize-to data/ka --docs 2000000
    python lm.py --data-dir data/ka --steps 100000

On TPU or a multi-GPU host the batch is sharded over every local device and
compute can drop to bf16 while the weights stay fp32:
    python lm.py --steps 20000 --docs 200000 --batch-size 128 --dtype bfloat16

Add --remat to trade ~4% more compute for a large cut in activation memory when
the batch you want will not otherwise fit.

Interrupted runs continue with their Adam moments *and* their exact position in
the data stream, so a resumed run matches an uninterrupted one:
    python lm.py --steps 20000 --resume
"""

import argparse
import dataclasses
import functools
import json
import math
import os
import time
from typing import Iterable, Iterator

import grain
import grain.experimental
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

PAD_ID, MASK_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]"]
N_SPECIAL = len(SPECIAL_TOKENS)

# Shortest window worth keeping. Below this a row is mostly [PAD] and carries
# too little context to predict from; above it, tails are worth their padding.
MIN_WINDOW_TOKENS = 32
# Held-out windows are sized in batches, not as a raw fraction: one eval batch
# makes the number pure noise.
MIN_VAL_BATCHES = 4

# Flipped off by --no-progress. Module-level because every phase draws through
# bar() and threading a flag into each of them would be noise.
PROGRESS = True


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    """Frozen so NNX can treat it as a static attribute of the modules."""

    vocab_size: int = 32_000
    hidden: int = 384
    layers: int = 12
    heads: int = 6
    mlp_hidden: int = 1024  # ~2.67x hidden, matches a 4x vanilla MLP's params
    max_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0  # ModernBERT trains with 0.0; raise for tiny corpora
    init_std: float = 0.02
    # Compute and storage dtypes are separate on purpose: bf16 compute with fp32
    # master weights is the standard TPU recipe, and folding them into one field
    # would put Adam's moments in bf16 along with the parameters.
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    # Recompute every block's activations in the backward pass. Off by default;
    # this is what you reach for when you want a batch the card will not hold.
    remat: bool = False

    @property
    def head_dim(self) -> int:
        assert self.hidden % self.heads == 0
        return self.hidden // self.heads


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


# ----------------------------------------------------------------------------
# Optimizer and schedule
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Device placement
# ----------------------------------------------------------------------------


def make_shardings(batch_size: int):
    """Single-host data parallelism: batch split over devices, params replicated.

    Returns (data_sharding, replicated) — or (None, None) on one device, so the
    single-accelerator path stays free of any sharding machinery.
    """
    n = jax.device_count()
    if n == 1:
        return None, None
    if batch_size % n:
        raise RuntimeError(
            f"--batch-size {batch_size} is not divisible by {n} devices; "
            f"uneven sharding fails deep inside pjit with a much worse message"
        )
    mesh = jax.make_mesh((n,), ("data",))
    return (jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))


def replicate(model, optimizer, repl) -> None:
    """Commit params and optimizer state to every device, once, up front.

    Sharding only the batch would leave these uncommitted on one device; GSPMD
    then infers replication and returns multi-device-committed outputs, so step
    two sees different input shardings than step one and recompiles.
    """
    if repl is None:
        return
    nnx.update(model, jax.device_put(nnx.state(model), repl))
    nnx.update(optimizer, jax.device_put(nnx.state(optimizer), repl))


def to_device(batch: dict, sharding):
    """Host numpy -> device arrays, sharded along the batch axis when asked."""
    if sharding is None:
        return {k: jnp.asarray(v) for k, v in batch.items()}
    return {k: jax.device_put(v, sharding) for k, v in batch.items()}


# nnx.jit adopts each argument's own sharding when the array is committed, so no
# in_shardings is needed here — device_put in to_device() does the committing.
# Donation lets XLA alias the input param/moment buffers into the outputs; it is
# a no-op on CPU and a real memory win on TPU/GPU.
@nnx.jit(donate_argnames=("model", "optimizer"))
def train_step(model: MlmEncoder, optimizer: nnx.Optimizer, batch):
    def loss_fn(m):
        return mlm_loss(m, batch)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: MlmEncoder, batch):
    return mlm_loss(model, batch)


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


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------


def save_config(save_dir: str, cfg: EncoderConfig) -> None:
    """The config is not part of the orbax tree, and without it there is nothing
    to rebuild the model into before restoring."""
    d = dataclasses.asdict(cfg)
    for k in ("dtype", "param_dtype"):
        d[k] = getattr(cfg, k).__name__
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(d, f, indent=2)


def load_config(save_dir: str) -> EncoderConfig:
    with open(os.path.join(save_dir, "config.json")) as f:
        d = json.load(f)
    # param_dtype is absent from configs written before it was split out; the
    # dataclass default (fp32) is the right answer for those.
    for k in ("dtype", "param_dtype"):
        if k in d:
            d[k] = getattr(jnp, d[k])
    return EncoderConfig(**d)


def checkpoint_manager(save_dir: str, keep: int = 10):
    import orbax.checkpoint as ocp

    os.makedirs(save_dir, exist_ok=True)
    return ocp.CheckpointManager(
        os.path.abspath(save_dir),  # orbax refuses relative paths
        options=ocp.CheckpointManagerOptions(max_to_keep=keep, create=True),
    )


def assert_writable(save_dir: str) -> None:
    """Refuse to train into a directory that already holds checkpoints.

    A fresh CheckpointManager over a populated directory silently *drops* every
    save below the highest step already present — it returns False and writes
    nothing. A run started this way looks healthy for hours and then has no
    checkpoints to show for it, so fail here instead.
    """
    if not os.path.isdir(save_dir):
        return
    mgr = checkpoint_manager(save_dir)
    steps = mgr.all_steps()
    mgr.close()
    if steps:
        raise RuntimeError(
            f"{os.path.abspath(save_dir)} already holds checkpoints at steps {sorted(steps)}. "
            f"orbax would silently discard every save below step {max(steps)}. "
            f"Pass --resume to continue that run, or point --save-dir somewhere empty."
        )


def save_checkpoint(mgr, step: int, model: MlmEncoder, optimizer,
                    data_iter=None) -> None:
    """Params, optimizer state, *and* the input pipeline's position.

    The trapezoid plateau is only a free parameter if a fork can resume Adam's
    moments, and a resumed run is only equivalent to an uninterrupted one if the
    data stream picks up where it stopped rather than reseeding into a different
    permutation.
    """
    import orbax.checkpoint as ocp

    items = {
        "model": ocp.args.StandardSave(nnx.state(model, nnx.Param)),
        "opt": ocp.args.StandardSave(nnx.state(optimizer)),
    }
    if data_iter is not None:
        items["data_iter"] = grain.checkpoint.CheckpointSave(item=data_iter)
    mgr.save(step, args=ocp.args.Composite(**items))


def restore_data_iter(save_dir: str, data_iter, step: int) -> None:
    """Restore the grain iterator in place, if this checkpoint carries one.

    Checkpoints written before the pipeline moved to grain have no `data_iter`
    entry; those still resume their weights, they just restart the stream.
    """
    import orbax.checkpoint as ocp

    mgr = checkpoint_manager(save_dir)
    try:
        mgr.restore(step, args=ocp.args.Composite(
            data_iter=grain.checkpoint.CheckpointRestore(item=data_iter)))
        print(f"  data pipeline resumed at its saved position")
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"  no data_iter in checkpoint {step} ({type(e).__name__}); "
              f"restarting the stream")
    finally:
        mgr.close()


def load_checkpoint(save_dir: str, step: int | None = None):
    """Rebuild an inference-ready model from disk. Returns (model, cfg, step)."""
    import orbax.checkpoint as ocp

    cfg = load_config(save_dir)
    model = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    mgr = checkpoint_manager(save_dir)
    step = mgr.latest_step() if step is None else step
    if step is None:
        raise RuntimeError(f"no checkpoints found in {save_dir}")
    restored = mgr.restore(step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(nnx.state(model, nnx.Param)),
    ))
    nnx.update(model, restored["model"])
    mgr.close()
    model.eval()
    return model, cfg, step


def resume_checkpoint(save_dir: str, schedule, step: int | None = None):
    """Rebuild model *and* optimizer, Adam moments included.

    StandardRestore restores into a target tree, so the optimizer has to be
    constructed first — with the same `tx`, which means the same schedule built
    from the same total step count. Resuming onto a different schedule would
    silently change the run, so main() checks that before calling this.

    Returns (model, optimizer, cfg, step).
    """
    import orbax.checkpoint as ocp

    cfg = load_config(save_dir)
    model = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    model.train()
    optimizer = make_optimizer(model, schedule)

    mgr = checkpoint_manager(save_dir)
    step = mgr.latest_step() if step is None else step
    if step is None:
        raise RuntimeError(f"no checkpoints found in {save_dir}")
    restored = mgr.restore(step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(nnx.state(model, nnx.Param)),
        opt=ocp.args.StandardRestore(nnx.state(optimizer)),
    ))
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["opt"])
    mgr.close()
    return model, optimizer, cfg, step


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

SMOKE_SENTENCES = [
    "ქართული ენა არის ქართველური ენების ჯგუფის ყველაზე გავრცელებული ენა.",
    "თბილისი საქართველოს დედაქალაქი და უდიდესი ქალაქია მდინარე მტკვრის ნაპირზე.",
    "ქართული დამწერლობა სამი ანბანისგან შედგება: ასომთავრული, ნუსხური და მხედრული.",
    "მთაწმინდა თბილისის ერთ-ერთი უბანია, სადაც მდებარეობს ტელევიზიის ანძა.",
    "ენის მოდელის წინასწარი ვარჯიში საჭიროებს დიდი რაოდენობის ტექსტურ მონაცემებს.",
]


def iter_texts(dataset: str, config: str, n_docs: int, smoke: bool) -> Iterator[str]:
    if smoke:
        for i in range(n_docs):
            yield " ".join(SMOKE_SENTENCES[(i + j) % len(SMOKE_SENTENCES)] for j in range(4))
        return

    from datasets import load_dataset

    stream = load_dataset(dataset, name=config, split="train", streaming=True)
    for i, row in enumerate(stream):
        if i >= n_docs:
            break
        text = row.get("text") or ""
        if len(text) > 200:
            yield text


def build_tokenizer(texts: Iterable[str], vocab_size: int, path: str):
    """Byte-level BPE trained on the target language only."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if os.path.exists(path):
        cached = Tokenizer.from_file(path)
        n = cached.get_vocab_size()
        # A cache smaller than requested is normal — the corpus may not support
        # the full budget. A cache *larger* than requested means --vocab-size was
        # lowered since it was built, and silently reusing it would train a model
        # with a different vocabulary than the flags describe.
        if n <= vocab_size:
            print(f"reusing tokenizer {path} (vocab {n})")
            return cached
        print(f"{path} has vocab {n} > requested {vocab_size}; retraining")

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        # Without this, only bytes that occur in the training corpus enter the
        # vocabulary and everything else falls back to [UNK] at inference — for
        # a Georgian corpus that silently guts Latin, digits and punctuation.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(path)
    return tok


class ByteTokenizer:
    """Zero-dependency fallback for --smoke: UTF-8 bytes offset past specials."""

    vocab = 256 + N_SPECIAL

    def encode_ids(self, text: str) -> list[int]:
        return [b + N_SPECIAL for b in text.encode("utf-8")]


def chunk_documents(texts: Iterable[str], tokenizer, seq_len: int, *,
                    with_doc_ids: bool = False, consume: bool = False,
                    progress=None):
    """One document per chunk boundary — never packs two documents together.

    NeoBERT measured cross-document sequence packing at -2.9 GLUE, so each
    window here stays inside a single document. Partial tails are padded.

    With `with_doc_ids`, also returns the source document index of every window,
    which is what lets the train/val split cut on document boundaries.

    With `consume`, pops each document off `texts` as it is tokenised, so the
    corpus of Python strings is released during the pass rather than staying
    live alongside the growing window table.
    """
    # Grow a preallocated array instead of stacking a list of rows at the end:
    # the list of ndarrays is slightly *larger* than the array it becomes, and
    # np.stack holds both at once.
    cap, n = 1024, 0
    rows = np.empty((cap, seq_len), dtype=np.int32)
    doc_ids = np.empty(cap, dtype=np.int32)

    for doc, ids in _encoded(texts, tokenizer, consume, progress=progress):
        for row in _windows(ids, seq_len):
            if n == cap:
                cap *= 2
                rows = np.resize(rows, (cap, seq_len))
                doc_ids = np.resize(doc_ids, cap)
            rows[n] = row
            doc_ids[n] = doc
            n += 1
    if not n:
        raise RuntimeError("no chunks produced — corpus too small or all filtered")
    rows = rows[:n].copy()
    return (rows, doc_ids[:n].copy()) if with_doc_ids else rows


def _draining(texts):
    """enumerate(), but releases each document as it is handed over.

    At --docs 200000 the corpus is a gigabyte or more of Python strings, and it
    would otherwise stay resident through the whole chunking pass on top of the
    window table being built.
    """
    if not isinstance(texts, list):
        yield from enumerate(texts)
        return
    texts.reverse()  # so popping from the end walks forward through the corpus
    i = 0
    while texts:
        yield i, texts.pop()
        i += 1


def bar(total=None, desc: str = "", unit: str = "it", initial: int = 0):
    """A tqdm if progress is on and tqdm is importable, otherwise a silent stub.

    Everything that draws progress goes through here so the whole feature is one
    import away from being off, and --no-progress leaves the old plain output.
    """
    if PROGRESS:
        try:
            from tqdm.auto import tqdm

            return tqdm(total=total, desc=desc, unit=unit, initial=initial,
                        dynamic_ncols=True, smoothing=0.05)
        except ImportError:
            pass

    class _Stub:
        n = 0

        def update(self, k=1):
            self.n += k

        def set_postfix(self, **kw):
            pass

        def write(self, s):
            print(s, flush=True)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    return _Stub()


def _encoded(texts, tokenizer, consume: bool = False, batch_size: int = 1000,
             progress=None):
    """(document index, token ids), tokenised a batch at a time.

    encode_batch_fast hands the whole batch to Rust, which spreads it across
    every core, and skips the per-token character offsets and word ids that
    encode() builds and this file throws away. Measured against the old
    one-document-at-a-time loop on a 40M-character Georgian corpus: 5.7x on ten
    cores, 2.2x on two — and 1.6x of that is the offset-free path rather than
    the threading, so the win survives even on a single core. The ids are
    byte-identical either way.

    batch_size is a memory dial, not a speed one: 256 through 20000 all measured
    within noise of each other, while peak RSS doubled at the top end.
    """
    source = _draining(texts) if consume else enumerate(texts)
    tick = (lambda k: None) if progress is None else progress.update

    if hasattr(tokenizer, "encode_ids"):  # ByteTokenizer (--smoke): no batch API
        for doc, text in source:
            yield doc, tokenizer.encode_ids(text)
            tick(1)
        return

    batch: list[str] = []
    first = 0
    for doc, text in source:
        if not batch:
            first = doc
        batch.append(text)
        if len(batch) == batch_size:
            for k, enc in enumerate(tokenizer.encode_batch_fast(batch)):
                yield first + k, enc.ids
            tick(len(batch))
            batch = []
    if batch:
        for k, enc in enumerate(tokenizer.encode_batch_fast(batch)):
            yield first + k, enc.ids
        tick(len(batch))


def _windows(ids, seq_len: int):
    """Token ids -> padded [CLS] ... [SEP] rows of exactly seq_len.

    Shared by the in-memory chunker and the streaming tokenise pass so the two
    cannot drift apart in how they cut documents.
    """
    body = seq_len - 2  # room for [CLS] and [SEP]
    for start in range(0, len(ids), body):
        window = ids[start:start + body]
        # Only drop windows too short to carry context. The old threshold of
        # body // 4 — 127 tokens at seq_len 512 — discarded every short
        # document outright and roughly a quarter of all document tails.
        if len(window) < MIN_WINDOW_TOKENS:
            continue
        row = np.full(seq_len, PAD_ID, dtype=np.int32)
        row[0] = CLS_ID
        row[1:1 + len(window)] = window
        row[1 + len(window)] = SEP_ID
        yield row


def split_by_document(doc_ids, val_frac: float, batch_size: int, seed: int):
    """Held-out mask over windows, cut on document boundaries.

    Splitting on windows instead would drop two chunks of the same article on
    opposite sides and the held-out loss would read back part of the training
    set.

    Uses the same per-document hash as the streaming pass, deliberately: the
    in-memory and --data-dir paths then produce byte-identical splits for the
    same corpus and seed, so a run can move between them without the held-out
    number shifting underneath it.
    """
    n_docs = int(doc_ids.max()) + 1
    is_val = np.fromiter((split_hash(int(d), seed) < val_frac for d in doc_ids),
                         dtype=bool, count=len(doc_ids))
    n_val = len({int(d) for d in doc_ids[is_val]})
    want = batch_size * MIN_VAL_BATCHES
    if int(is_val.sum()) < want:
        raise RuntimeError(
            f"only {int(is_val.sum())} held-out windows from {n_val} documents; "
            f"need {want} ({MIN_VAL_BATCHES} batches of {batch_size}). "
            f"Raise --val-frac above {val_frac}, or lower --batch-size."
        )
    return is_val, n_val, n_docs


def split_hash(doc: int, seed: int) -> float:
    """A deterministic uniform draw in [0, 1) keyed on the document index.

    The streaming pass never has every document id in hand at once, so the
    train/val decision has to be a pure function of the index rather than a
    permutation. This is a splitmix-style 32-bit mixer — uniform enough for a
    split, and it costs nothing.
    """
    x = (doc * 0x9E3779B1 + seed * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x / 4294967296.0


def tokenize_to(args, out_dir: str) -> None:
    """One-time CPU pass: corpus -> train/val ArrayRecord files + meta.json.

    Streams throughout. Nothing accumulates in host RAM but the tokenizer's
    training sample, so --docs is bounded by disk rather than by memory, and a
    GPU or TPU session never spends its time on this.
    """
    from array_record.python import array_record_module

    os.makedirs(out_dir, exist_ok=True)
    timings: dict[str, float] = {}

    # Pass 1: a prefix, only to fit the vocabulary. A 32k byte-level BPE is
    # converged long before a million documents, and training scales linearly,
    # so feeding it the whole corpus costs minutes for a vocabulary that does
    # not differ.
    t0 = time.time()
    if args.smoke:
        tokenizer, vocab_size = ByteTokenizer(), ByteTokenizer.vocab
        sample = list(iter_texts(args.dataset, args.config, args.docs, True))
        timings["sample"] = time.time() - t0
    else:
        n_sample = min(args.tokenizer_docs, args.docs)
        sample = []
        pb = bar(total=n_sample, desc="1/3 sample", unit="doc")
        for text in iter_texts(args.dataset, args.config, n_sample, False):
            sample.append(text)
            pb.update(1)
        pb.close()
        timings["sample"] = time.time() - t0

        t0 = time.time()
        print(f"training a {args.vocab_size}-token BPE on {len(sample)} documents ...",
              flush=True)
        tokenizer = build_tokenizer(sample, args.vocab_size, args.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        timings["train tokenizer"] = time.time() - t0

    # Pass 2: re-stream, encode in batches, and write rows out as they appear.
    # When the sample already covers the whole corpus there is nothing to
    # re-download — drain it instead, which is the --smoke case and any run
    # with --docs at or below --tokenizer-docs.
    t0 = time.time()
    reuse = len(sample) >= args.docs
    source = sample if reuse else iter_texts(args.dataset, args.config, args.docs, args.smoke)
    if not reuse:
        del sample

    paths = {k: os.path.join(out_dir, f"{k}.array_record") for k in ("train", "val")}
    # group_size is the decompression unit, and grain reads every source through
    # random access even when the consumer is sequential — it warns loudly at
    # anything above 1. So 1 for both files, val included.
    writers = {k: array_record_module.ArrayRecordWriter(v, "group_size:1")
               for k, v in paths.items()}
    rows_of = {"train": 0, "val": 0}
    docs_of = {"train": 0, "val": 0}
    pb = bar(total=args.docs, desc="2/3 encode", unit="doc")
    try:
        for doc, ids in _encoded(source, tokenizer, consume=reuse, progress=pb):
            side = "val" if split_hash(doc, args.seed) < args.val_frac else "train"
            docs_of[side] += 1
            for row in _windows(ids, args.seq_len):
                writers[side].write(row.astype("<i4").tobytes())
                rows_of[side] += 1
    finally:
        pb.close()
        for w in writers.values():
            w.close()  # required: the chunk index is written here
    timings["encode + write"] = time.time() - t0

    want = args.batch_size * MIN_VAL_BATCHES
    if rows_of["val"] < want:
        raise RuntimeError(
            f"only {rows_of['val']} held-out windows from {docs_of['val']} documents; "
            f"need {want} ({MIN_VAL_BATCHES} batches of {args.batch_size}). "
            f"Raise --val-frac above {args.val_frac}, or lower --batch-size."
        )

    n_docs = docs_of["train"] + docs_of["val"]
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"seq_len": args.seq_len, "vocab_size": vocab_size,
                   "tokenizer": None if args.smoke else args.tokenizer_path,
                   "n_train": rows_of["train"], "n_val": rows_of["val"],
                   "n_docs": n_docs, "val_frac": args.val_frac,
                   "seed": args.seed}, f, indent=2)

    total = sum(timings.values()) or 1e-9
    print(f"wrote {rows_of['train']} train / {rows_of['val']} val windows of "
          f"{args.seq_len} tokens ({docs_of['val']}/{n_docs} documents held out), "
          f"vocab {vocab_size}")
    print("  timing: " + "  ".join(f"{k} {v:.1f}s ({v / total:.0%})"
                                   for k, v in timings.items()))
    print(f"  -> {os.path.abspath(out_dir)}; train with --data-dir {out_dir}")


def read_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.json")) as f:
        return json.load(f)


def n_predictions(seq_len: int, mask_prob: float) -> int:
    """Target slots per row — BERT's max_predictions_per_seq.

    Fixed for the run so the gather in MlmEncoder.__call__ is statically shaped.
    """
    return max(1, int(round(mask_prob * seq_len)))


def mask_batch(rows: np.ndarray, rng: np.random.Generator, mask_prob: float,
               n_pred: int):
    """Dynamic masking, regenerated every time (RoBERTa) rather than per corpus.

    30% of maskable positions -> [MASK] 100% of the time. No 80/10/10 split.

    Returns the targets *gathered* into [B, n_pred] rather than scattered across
    [B, seq_len] with -100 padding: only these positions are ever projected into
    the vocabulary, which is roughly a third of the logits and a quarter of the
    step. Rows with fewer maskable tokens than n_pred get weight 0 in the tail.
    """
    rows = np.atleast_2d(rows)
    attention_mask = (rows != PAD_ID).astype(np.int32)
    maskable = attention_mask.astype(bool) & (rows >= N_SPECIAL)

    # rank maskable positions by a random key and take the first n_pred; this
    # replaces both the per-position bernoulli draw and the per-row loop that
    # used to guarantee at least one target
    keys = rng.random(rows.shape)
    keys[~maskable] = np.inf
    order = np.argsort(keys, axis=1, kind="stable")[:, :n_pred]

    n_maskable = maskable.sum(1, keepdims=True)
    k = np.clip(np.rint(mask_prob * n_maskable), 1, None)
    k = np.minimum(k, np.minimum(n_maskable, n_pred))
    keep = np.arange(n_pred)[None, :] < k  # [B, n_pred]

    chosen = np.take_along_axis(rows, order, axis=1)
    input_ids = rows.copy()  # `rows` may be a view onto the chunk table
    np.put_along_axis(input_ids, order, np.where(keep, MASK_ID, chosen), axis=1)
    return {
        "input_ids": input_ids.astype(np.int32),
        "attention_mask": attention_mask,
        "positions": np.where(keep, order, 0).astype(np.int32),
        "targets": np.where(keep, chosen, 0).astype(np.int32),
        "weights": keep.astype(np.float32),
    }


class MaskExample(grain.transforms.RandomMap):
    """Per-example dynamic masking.

    Grain derives this element's RNG from its index by resetting a Philox
    counter, so the mask a given window receives is a pure function of that
    index. Two consequences worth having: changing --batch-size no longer
    changes every mask, and a resumed run reproduces masks exactly rather than
    relying on the RNG state having been checkpointed.
    """

    def __init__(self, mask_prob: float, n_pred: int):
        self.mask_prob = mask_prob
        self.n_pred = n_pred

    def random_map(self, row, rng):
        out = mask_batch(row, rng, self.mask_prob, self.n_pred)
        return {k: v[0] for k, v in out.items()}  # drop the batch axis


def decode_record(raw: bytes) -> np.ndarray:
    """One ArrayRecord payload -> one window.

    The astype is not redundant: np.frombuffer returns a read-only view over
    immutable bytes, and a writeable copy costs 2 KB here.
    """
    return np.frombuffer(raw, dtype="<i4").astype(np.int32)


def make_dataset(source, batch_size: int, mask_prob: float, n_pred: int,
                 seed: int, *, decode: bool = False, repeat: bool = True):
    """The whole input pipeline.

    `source` is either the in-memory [N, seq_len] chunk table (a 2-D ndarray
    already satisfies grain's RandomAccessDataSource protocol) or an
    ArrayRecordDataSource, in which case `decode` turns bytes back into rows.

    Deliberately no mp_prefetch: grain spawns workers and ships the dataset
    graph through cloudpickle, which would capture the source array itself and
    give every worker a full resident copy with no copy-on-write. On an
    in-memory source that is a much larger memory problem than the one this
    pipeline solves. num_threads=0 for the same reason grain's own docs give —
    the data is already in RAM, so reader threads only contend on the GIL.
    """
    ds = grain.MapDataset.source(source)
    if decode:
        ds = ds.map(decode_record)
    ds = ds.seed(seed).shuffle()
    if repeat:
        ds = ds.repeat()
    return (
        ds.random_map(MaskExample(mask_prob, n_pred))
        .batch(batch_size, drop_remainder=True)
        .to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=64))
    )


def device_stream(iter_ds, sharding):
    """Overlap host masking with device compute, and double-buffer on device."""
    target = sharding if sharding is not None else jax.devices()[0]
    return grain.experimental.device_put(
        iter_ds, target, cpu_buffer_size=4, device_buffer_size=2)


def evaluate(model, val_ds, max_batches: int = 32):
    """Mean MLM loss over held-out windows.

    No re-seeding needed any more: masks are a pure function of element index,
    so two evals over the same finite dataset are comparable by construction.
    Capped at max_batches so eval cost stays flat as the split grows.
    """
    losses = []
    for batch in val_ds:
        if len(losses) >= max_batches:
            break
        losses.append(eval_step(model, batch))
    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))  # one sync, not one per batch


# ----------------------------------------------------------------------------
# Self-test — the invariants worth asserting before a long run
# ----------------------------------------------------------------------------


def selftest():
    cfg = EncoderConfig(vocab_size=500, hidden=64, layers=2, heads=4,
                        mlp_hidden=128, max_len=16)
    model = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    model.eval()
    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(N_SPECIAL, 500, (1, 16)), jnp.int32)
    mask = jnp.ones((1, 16), jnp.int32)
    # the same sequence, but with positions 12..15 declared padding
    padded = jnp.concatenate(
        [jnp.ones((1, 12), jnp.int32), jnp.zeros((1, 4), jnp.int32)], axis=1
    )
    alt = ids.at[0, 12].set(77)

    def changed(attention_mask):
        a = model.encode(ids, attention_mask)
        b = model.encode(alt, attention_mask)
        return float(jnp.abs(a[0, 3] - b[0, 3]).max()) > 1e-6

    assert changed(mask), "position 3 ignores position 12 — attention is not bidirectional"
    assert not changed(padded), "position 12 is padded but still reaches position 3 — pad mask is broken"

    # `changed` only watches information travel backwards, from 12 to 3. An
    # anti-causal model — every position attending to later ones and nothing
    # else — satisfies it while being no more bidirectional than a GPT.
    base = model.encode(ids, mask)
    early = model.encode(ids.at[0, 1].set(88), mask)
    assert float(jnp.abs(base[0, 12] - early[0, 12]).max()) > 1e-6, \
        "position 12 ignores position 1 — attention only flows backward"

    # The RoPE asserts below test apply_rope in isolation, which says nothing
    # about whether the model calls it. Deleting both apply_rope lines from
    # BidirectionalAttention leaves every one of them passing and yields a bag
    # of words: permuting tokens then leaves the attended set unchanged.
    perm = ids.at[0, 3:6].set(ids[0, 3:6][::-1])
    assert float(jnp.abs(base[0, 0] - model.encode(perm, mask)[0, 0]).max()) > 1e-6, \
        "reordering tokens changed nothing — bag of words; RoPE is not wired into attention"

    # RoPE encodes *relative* position: with q and k held constant across
    # positions, the score q_i . k_j must depend only on the offset j - i, and
    # must actually vary with it. Both halves are load-bearing — a no-op
    # apply_rope is trivially "relative", and an absolute scheme does vary.
    cos, sin = rope_tables(16, 8, 10_000.0, jnp.float32)
    everywhere = lambda: jnp.broadcast_to(
        jnp.asarray(rng.standard_normal((1, 1, 1, 8)), jnp.float32), (1, 1, 16, 8)
    )
    qr = apply_rope(everywhere(), cos, sin)[0, 0]  # [16, 8]
    kr = apply_rope(everywhere(), cos, sin)[0, 0]
    scores = qr @ kr.T  # scores[i, j] = <rope(q, i), rope(k, j)>
    diags = [jnp.diagonal(scores, off) for off in range(-8, 9)]
    drift = max(float(d.max() - d.min()) for d in diags)
    spread = float(max(d.mean() for d in diags) - min(d.mean() for d in diags))
    assert drift < 1e-4, f"score varies within a fixed offset ({drift:.2e}) — RoPE is not relative"
    assert spread > 1e-2, f"score flat across offsets ({spread:.2e}) — apply_rope is a no-op"

    # a correctly initialised MLM head starts at ln(vocab_size)
    full = MlmEncoder(EncoderConfig(), rngs=nnx.Rngs(0))
    rows = rng.integers(N_SPECIAL, 32_000, (2, 128)).astype(np.int32)
    n_pred = n_predictions(128, 0.30)
    batch = {k: jnp.asarray(v)
             for k, v in mask_batch(rows, rng, 0.30, n_pred).items()}
    loss = float(mlm_loss(full, batch))
    # Bounded loosely on purpose. The expected value is ln(vocab) + sigma^2 / 2,
    # where sigma is the init logit spread, so the true centre sits just above
    # ln(vocab) and moves with the seed — a tight two-sided band flakes.
    assert abs(loss - math.log(32_000)) < 0.4, f"init loss {loss:.3f} != ln(vocab)"

    # mask_batch's contract, on a row that actually contains CLS/SEP/PAD — the
    # rows above are pure content, so none of this is otherwise exercised. A
    # mask_batch that forgot to mask at all passes every assert before this one.
    row = np.concatenate([[CLS_ID], rng.integers(N_SPECIAL, 500, 10), [SEP_ID], np.zeros(5, int)])
    rows_c = np.stack([row, row]).astype(np.int32)
    mb = mask_batch(rows_c, rng, 0.30, n_predictions(rows_c.shape[1], 0.30))
    pos, kept = mb["positions"], mb["weights"] > 0
    at_pos = np.take_along_axis(rows_c, pos, axis=1)
    assert (at_pos[kept] >= N_SPECIAL).all(), "a special or PAD token was chosen as a target"
    assert np.array_equal(mb["targets"][kept], at_pos[kept]), \
        "targets do not match the tokens at the gathered positions"
    assert (np.take_along_axis(mb["input_ids"], pos, axis=1)[kept] == MASK_ID).all(), \
        "a scored position was not replaced by [MASK] in the input"
    # ...and nothing else in the input moved
    expected = np.zeros_like(rows_c, dtype=bool)
    np.put_along_axis(expected, pos, kept, axis=1)
    assert np.array_equal(mb["input_ids"] != rows_c, expected), \
        "input_ids changed somewhere other than the scored positions"
    assert np.array_equal(mb["attention_mask"], (rows_c != PAD_ID).astype(np.int32)), "attention_mask is wrong"

    # The guard on M1: gathering the scored positions must not change the
    # objective. Scatter the targets back out and score every position the old
    # way; the two losses must agree.
    lab = np.full(rows.shape, -100, np.int32)
    np.put_along_axis(lab, np.asarray(batch["positions"]),
                      np.where(np.asarray(batch["weights"]) > 0,
                               np.asarray(batch["targets"]), -100), axis=1)
    lab = jnp.asarray(lab)
    dense = full(batch["input_ids"], batch["attention_mask"])  # [B, L, vocab]
    ce = optax.softmax_cross_entropy_with_integer_labels(
        dense.astype(jnp.float32), jnp.where(lab >= 0, lab, 0))
    wt = (lab >= 0).astype(jnp.float32)
    dense_loss = float((ce * wt).sum() / jnp.maximum(wt.sum(), 1.0))
    assert abs(dense_loss - loss) < 1e-4, \
        f"gathered loss {loss:.6f} != full-width loss {dense_loss:.6f} — the gather changed the objective"

    # The optimizer path is otherwise never executed here. A few steps on one
    # fixed batch must drive the loss down; if they do not, gradients are not
    # reaching the parameters and no amount of pretraining will help.
    small = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    small.train()
    opt = make_optimizer(small, trapezoid_schedule(1e-3, 40))
    fixed = to_device(mask_batch(rng.integers(N_SPECIAL, 500, (2, 16)).astype(np.int32),
                                 rng, 0.30, n_predictions(16, 0.30)), None)
    first = float(train_step(small, opt, fixed))
    for _ in range(39):
        last = float(train_step(small, opt, fixed))
    assert last < first - 0.1, f"40 steps on one batch moved the loss {first:.3f} -> {last:.3f}"

    # Same seed, same number — otherwise a run is not reproducible and two
    # evals are not comparable.
    twice = [float(mlm_loss(MlmEncoder(cfg, rngs=nnx.Rngs(7)), fixed)) for _ in range(2)]
    assert twice[0] == twice[1], f"same seed gave {twice[0]} then {twice[1]}"

    counts = count_params(full)
    print("self-test passed")
    print(f"  bidirectional both ways, position in use, pad mask, RoPE relativity")
    print(f"  mask contract, optimizer ({first:.2f} -> {last:.2f}), determinism, init loss {loss:.3f}")
    print(f"  default config: {counts['total'] / 1e6:.2f}M params "
          f"({counts['embedding'] / 1e6:.2f}M of it the tied table)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny synthetic run, no network")
    p.add_argument("--selftest", action="store_true", help="assert the architecture invariants and exit")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    p.add_argument("--config", default="kat_Geor")
    p.add_argument("--docs", type=int, default=20_000)
    p.add_argument("--tokenizer-path", default="ka_bpe.json")
    p.add_argument("--vocab-size", type=int, default=32_000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--peak-lr", type=float, default=6e-4)
    p.add_argument("--mask-prob", type=float, default=0.30)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="compute dtype; parameters and Adam moments stay float32")
    p.add_argument("--val-frac", type=float, default=0.01,
                   help="fraction of *documents* held out of training for eval")
    p.add_argument("--eval-every", type=int, default=500,
                   help="held-out eval interval; 0 to evaluate only at the end")
    p.add_argument("--save-dir", default="checkpoints",
                   help="orbax checkpoint directory; pass an empty string to disable")
    p.add_argument("--save-every", type=int, default=500)
    # Keep enough that some survive on the plateau. At --save-every 500 a 20k
    # step run with --keep 3 retains only checkpoints inside the decay ramp,
    # which is exactly what the trapezoid schedule exists to let you fork from.
    p.add_argument("--keep", type=int, default=10, help="checkpoints to retain")
    p.add_argument("--resume", action="store_true",
                   help="continue the run in --save-dir, Adam moments included")
    p.add_argument("--remat", action="store_true",
                   help="recompute block activations in the backward pass: large "
                        "memory saving for roughly 4%% more compute")
    p.add_argument("--tokenizer-docs", type=int, default=200_000,
                   help="documents used to fit the BPE; a 32k vocab is converged "
                        "long before this, and more only costs time")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="live progress bars; --no-progress restores plain periodic lines")
    p.add_argument("--tokenize-to", default="",
                   help="write train/val ArrayRecord files to this directory and exit")
    p.add_argument("--data-dir", default="",
                   help="train from ArrayRecord files written by --tokenize-to, "
                        "instead of building the corpus in memory")
    args = p.parse_args()

    global PROGRESS
    PROGRESS = args.progress

    print(f"smoke test: {args.smoke} ...")

    if args.selftest:
        selftest()
        return

    if args.smoke:
        # Only override what the caller left at its default, so a flag passed
        # explicitly alongside --smoke still means what it says.
        for name, value in [("docs", 200), ("seq_len", 128), ("steps", 30),
                            ("batch_size", 8), ("layers", 4), ("hidden", 128),
                            ("heads", 4), ("log_every", 5), ("eval_every", 10),
                            ("save_every", 15), ("val_frac", 0.05)]:
            if getattr(args, name) == p.get_default(name):
                setattr(args, name, value)

    if args.tokenize_to:
        tokenize_to(args, args.tokenize_to)
        return

    if args.data_dir:
        # Pre-tokenised: windows stream off disk, so host RAM no longer caps the
        # corpus and this session spends none of its time tokenising.
        meta = read_meta(args.data_dir)
        if meta["seq_len"] != args.seq_len:
            raise RuntimeError(
                f"{args.data_dir} was written at --seq-len {meta['seq_len']}, "
                f"but this run asks for {args.seq_len}"
            )
        vocab_size, decode = meta["vocab_size"], True
        train_source = grain.sources.ArrayRecordDataSource(
            os.path.join(args.data_dir, "train.array_record"))
        val_source = grain.sources.ArrayRecordDataSource(
            os.path.join(args.data_dir, "val.array_record"))
        print(f"corpus: {len(train_source)} train / {len(val_source)} val windows "
              f"of {args.seq_len} tokens from {args.data_dir}, vocab {vocab_size}")
    else:
        print("building corpus ...", flush=True)
        texts = list(iter_texts(args.dataset, args.config, args.docs, args.smoke))
        tokenizer = (ByteTokenizer() if args.smoke
                     else build_tokenizer(texts, args.vocab_size, args.tokenizer_path))
        vocab_size = (ByteTokenizer.vocab if args.smoke else tokenizer.get_vocab_size())

        n_texts = len(texts)
        pb = bar(total=n_texts, desc="encode", unit="doc")
        chunks, doc_ids = chunk_documents(texts, tokenizer, args.seq_len,
                                          with_doc_ids=True, consume=True, progress=pb)
        pb.close()
        print(f"corpus: {n_texts} docs -> {len(chunks)} sequences of {args.seq_len} tokens "
              f"({len(chunks) * args.seq_len / 1e6:.2f}M tokens), vocab {vocab_size}")
        del texts  # already drained by consume=True; drop the empty list too

        is_val, n_val_docs, n_docs = split_by_document(
            doc_ids, args.val_frac, args.batch_size, args.seed)
        train_source, val_source, decode = chunks[~is_val], chunks[is_val], False
        # the fancy-indexed halves above are copies; without this the whole
        # corpus stays resident twice for the life of the run
        del chunks, doc_ids, is_val
        if len(val_source) < args.batch_size or len(train_source) < args.batch_size:
            raise RuntimeError(
                f"corpus too small to split: {n_docs} docs, batch {args.batch_size}"
            )
        print(f"split: {len(train_source)} train / {len(val_source)} held-out windows "
              f"({n_val_docs}/{n_docs} documents held out)")

    # Built before the model because resuming has to rebuild the optimizer with
    # the *same* tx, and the schedule is baked into it.
    schedule = trapezoid_schedule(args.peak_lr, args.steps)
    data_sharding, repl = make_shardings(args.batch_size)

    if args.resume:
        if not args.save_dir:
            raise RuntimeError("--resume needs a --save-dir to resume from")
        model, optimizer, cfg, start_step = resume_checkpoint(args.save_dir, schedule)
        if cfg.vocab_size != vocab_size:
            raise RuntimeError(
                f"checkpoint has vocab {cfg.vocab_size} but the tokenizer has {vocab_size}"
            )
        if start_step >= args.steps:
            raise RuntimeError(
                f"checkpoint is at step {start_step} and --steps is {args.steps}; "
                f"--steps is the *total* run length and defines the LR schedule, "
                f"so raise it to continue"
            )
        print(f"resumed from step {start_step} in {os.path.abspath(args.save_dir)}")
    else:
        if args.save_dir:
            assert_writable(args.save_dir)
        cfg = EncoderConfig(
            vocab_size=vocab_size,
            hidden=args.hidden,
            layers=args.layers,
            heads=args.heads,
            mlp_hidden=int(round(args.hidden * 8 / 3 / 64)) * 64,
            max_len=args.seq_len,
            dropout=args.dropout,
            dtype=getattr(jnp, args.dtype),
            remat=args.remat,
        )
        model = MlmEncoder(cfg, rngs=nnx.Rngs(args.seed))
        model.train()
        optimizer = make_optimizer(model, schedule)
        start_step = 0

    replicate(model, optimizer, repl)
    if repl is not None:
        print(f"data-parallel over {jax.device_count()} devices, "
              f"{args.batch_size // jax.device_count()} sequences each")

    counts = count_params(model)
    print(f"params: {counts['total'] / 1e6:.2f}M total, "
          f"{counts['non_embedding'] / 1e6:.2f}M non-embedding, "
          f"{counts['embedding'] / 1e6:.2f}M in the tied table "
          f"({cfg.dtype.__name__} compute, {cfg.param_dtype.__name__} params)")

    mgr = checkpoint_manager(args.save_dir, args.keep) if args.save_dir else None
    if mgr is not None and not args.resume:
        # Only on a fresh run: rewriting this on resume would clobber the config
        # that the checkpoints already in the directory were trained under.
        save_config(args.save_dir, cfg)

    n_pred = n_predictions(args.seq_len, args.mask_prob)
    train_ds = make_dataset(train_source, args.batch_size, args.mask_prob, n_pred,
                            args.seed + 1, decode=decode)
    val_ds = make_dataset(val_source, args.batch_size, args.mask_prob, n_pred,
                          args.seed + 2, decode=decode, repeat=False)
    data_iter = iter(device_stream(train_ds, data_sharding))
    if args.resume:
        # The whole point of checkpointing the iterator: restore the exact
        # position rather than reseeding into a different stream.
        restore_data_iter(args.save_dir, data_iter, start_step)

    tokens_per_step = args.batch_size * args.seq_len
    print(f"training steps {start_step + 1}..{args.steps} at {args.mask_prob:.0%} masking, "
          f"{tokens_per_step} tokens/step")

    start = time.time()
    pb = bar(total=args.steps, desc="train", unit="step", initial=start_step)
    for step in range(start_step + 1, args.steps + 1):
        batch = next(data_iter)
        loss = train_step(model, optimizer, batch)
        pb.update(1)
        if step % args.log_every == 0 or step == start_step + 1:
            # float(loss) is a device sync, so it stays on the log cadence
            # rather than happening every step just to feed the bar.
            loss = float(loss)
            elapsed = time.time() - start
            done = step - start_step  # this run's steps, not the schedule's
            rate = done * tokens_per_step / max(elapsed, 1e-6) / 1e3
            lr = float(schedule(step - 1))
            if PROGRESS:
                pb.set_postfix(loss=f"{loss:.3f}",
                               ppl=f"{math.exp(min(loss, 20)):.1f}",
                               lr=f"{lr:.2e}", tok_s=f"{rate:.1f}k")
            else:
                print(f"step {step:>6}  loss {loss:6.3f}  "
                      f"ppl {math.exp(min(loss, 20)):9.1f}  "
                      f"lr {lr:.2e}  {rate:6.1f}k tok/s", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            # `deterministic` is static, so toggling it compiles a second copy of
            # the whole 12-layer graph. At dropout 0.0 the two are identical, so
            # skip the toggle and the extra compile with it.
            if cfg.dropout > 0.0:
                model.eval()
            val = evaluate(model, device_stream(val_ds, data_sharding))
            if cfg.dropout > 0.0:
                model.train()
            pb.write(f"step {step:>6}  held-out loss {val:6.3f}  (train {float(loss):6.3f})")
        if mgr is not None and args.save_every and step % args.save_every == 0:
            save_checkpoint(mgr, step, model, optimizer, data_iter)

    pb.close()
    model.eval()
    final = evaluate(model, device_stream(val_ds, data_sharding))
    print(f"final held-out mlm loss {final:.3f}")

    if mgr is not None:
        save_checkpoint(mgr, args.steps, model, optimizer, data_iter)
        mgr.wait_until_finished()
        print(f"checkpoints in {os.path.abspath(args.save_dir)}: steps {mgr.all_steps()}")
        mgr.close()

    # embeddings for downstream use: mean-pool the hidden states over content
    # tokens only. `>= N_SPECIAL` drops [PAD], [CLS], [SEP] and [MASK] at once —
    # [CLS] in particular is an untrained attention sink here, so averaging it in
    # would fold a high-norm outlier into every embedding. Fed clean windows,
    # not masked ones: corrupting the input is a training device, not a step you
    # want between raw text and its vector.
    #
    # This is a smoke check that the encode path works, not evidence that the
    # vectors are good. MLM puts no loss on the aggregate of a sequence, so the
    # pooled space comes out anisotropic and similarity is dominated by token
    # frequency — a contrastive stage is what actually shapes it.
    rows = np.stack([np.asarray(val_source[i]) if not decode
                     else decode_record(val_source[i])
                     for i in range(args.batch_size)])
    batch = to_device({"ids": rows, "am": (rows != PAD_ID).astype(np.int32)}, data_sharding)
    hidden = nnx.jit(lambda m, i, a: m.encode(i, a))(model, batch["ids"], batch["am"])
    w = (batch["ids"] >= N_SPECIAL)[..., None].astype(hidden.dtype)
    pooled = (hidden * w).sum(1) / jnp.maximum(w.sum(1), 1.0)
    print(f"pooled embedding shape {pooled.shape}")


if __name__ == "__main__":
    main()