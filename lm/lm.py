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

On TPU or a multi-GPU host the batch is sharded over every local device and
compute can drop to bf16 while the weights stay fp32:
    python lm.py --steps 20000 --docs 200000 --batch-size 128 --dtype bfloat16

Interrupted runs continue with their Adam moments intact:
    python lm.py --steps 20000 --resume
"""

import argparse
import dataclasses
import json
import math
import os
import time
from typing import Iterable, Iterator

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

        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(d)
        scores = scores + bias
        # softmax in fp32 regardless of compute dtype
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
        weights = self.drop(weights)

        out = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
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

    def encode(self, input_ids, attention_mask):
        """Returns hidden states [B, L, hidden] — use this for embeddings."""
        cfg = self.cfg
        _, l = input_ids.shape
        cos, sin = rope_tables(l, cfg.head_dim, cfg.rope_theta, cfg.dtype)
        bias = attention_bias(attention_mask, cfg.dtype)

        x = self.embed_drop(self.tok.encode(input_ids))
        for block in self.blocks:
            x = block(x, bias, cos, sin)
        return self.final_norm(x)

    def __call__(self, input_ids, attention_mask):
        """Returns logits [B, L, vocab_size]."""
        x = self.encode(input_ids, attention_mask)
        return self.tok.decode(self.head(x))


# ----------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------


def _masked_ce(logits, labels, weights):
    """Cross-entropy averaged over positions where weights == 1.

    Computed over every position and then masked, rather than gathering the
    masked positions: gathering gives a batch-dependent shape and forces jax
    to retrace on every new count.
    """
    safe = jnp.where(labels >= 0, labels, 0)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, safe)
    return (ce * weights).sum() / jnp.maximum(weights.sum(), 1.0)


def mlm_loss(model: MlmEncoder, batch):
    logits = model(batch["input_ids"], batch["attention_mask"])
    labels = batch["labels"]
    return _masked_ce(logits, labels, (labels >= 0).astype(jnp.float32))


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


def save_checkpoint(mgr, step: int, model: MlmEncoder, optimizer) -> None:
    """Params *and* optimizer state. The trapezoid plateau is only a free
    parameter if a fork can resume Adam's moments, not just the weights.
    """
    import orbax.checkpoint as ocp

    mgr.save(step, args=ocp.args.Composite(
        model=ocp.args.StandardSave(nnx.state(model, nnx.Param)),
        opt=ocp.args.StandardSave(nnx.state(optimizer)),
    ))


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
                    with_doc_ids: bool = False):
    """One document per chunk boundary — never packs two documents together.

    NeoBERT measured cross-document sequence packing at -2.9 GLUE, so each
    window here stays inside a single document. Partial tails are padded.

    With `with_doc_ids`, also returns the source document index of every window,
    which is what lets the train/val split cut on document boundaries.
    """
    body = seq_len - 2  # room for [CLS] and [SEP]
    chunks: list[np.ndarray] = []
    doc_ids: list[int] = []
    for doc, text in enumerate(texts):
        if hasattr(tokenizer, "encode_ids"):
            ids = tokenizer.encode_ids(text)
        else:
            ids = tokenizer.encode(text).ids
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
            chunks.append(row)
            doc_ids.append(doc)
    if not chunks:
        raise RuntimeError("no chunks produced — corpus too small or all filtered")
    rows = np.stack(chunks)
    return (rows, np.asarray(doc_ids, dtype=np.int64)) if with_doc_ids else rows


def mask_batch(rows: np.ndarray, rng: np.random.Generator, mask_prob: float):
    """Dynamic masking, regenerated per batch (RoBERTa) rather than per corpus.

    30% of maskable positions -> [MASK] 100% of the time. No 80/10/10 split.
    """
    attention_mask = (rows != PAD_ID).astype(np.int32)
    maskable = attention_mask.astype(bool) & (rows >= N_SPECIAL)

    selected = (rng.random(rows.shape) < mask_prob) & maskable
    # guarantee at least one target per row, or the loss is undefined for it
    for i in np.where(~selected.any(axis=1))[0]:
        candidates = np.flatnonzero(maskable[i])
        if candidates.size:
            selected[i, rng.choice(candidates)] = True

    # `rows` is a fancy-indexed copy of the chunk table and is never mutated
    labels = np.where(selected, rows, -100).astype(np.int32)
    input_ids = np.where(selected, MASK_ID, rows).astype(np.int32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def batch_stream(chunks, batch_size, mask_prob, seed=0, sharding=None):
    rng = np.random.default_rng(seed)
    n = len(chunks)
    while True:
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            rows = chunks[order[i:i + batch_size]]
            yield to_device(mask_batch(rows, rng, mask_prob), sharding)


def prefetch(batches, depth: int = 2):
    """Run the host-side masking a few batches ahead of the device.

    mask_batch is numpy on the CPU and the transfer that follows it is not free.
    Without this the accelerator sits idle through both on every single step —
    invisible on CPU, a real bubble on TPU. The thread is a daemon because
    batch_stream never terminates.
    """
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def fill():
        try:
            for b in batches:
                q.put(b)
        finally:
            q.put(done)

    threading.Thread(target=fill, daemon=True).start()
    while True:
        b = q.get()
        if b is done:
            return
        yield b


def evaluate(model, chunks, batch_size, mask_prob, seed=1234, max_batches=32,
             sharding=None):
    """Mean MLM loss over held-out windows.

    The rng is re-seeded on every call so the masking pattern is identical each
    time: otherwise the number moves with the mask draw and two evals are not
    comparable. Capped at max_batches so eval cost stays flat as the split grows.
    """
    rng = np.random.default_rng(seed)
    total, n = 0.0, 0
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        if n >= max_batches:
            break
        rows = chunks[i:i + batch_size]
        total += float(eval_step(model, to_device(mask_batch(rows, rng, mask_prob), sharding)))
        n += 1
    return total / max(n, 1)


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
    batch = {k: jnp.asarray(v)
             for k, v in mask_batch(rows, rng, 0.30).items()}
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
    mb = mask_batch(rows_c, rng, 0.30)
    assert np.array_equal(mb["labels"] >= 0, mb["input_ids"] == MASK_ID), \
        "labels disagree with [MASK] positions — the target set and the corruption have drifted apart"
    assert (mb["labels"][mb["labels"] >= 0] >= N_SPECIAL).all(), "a special token was chosen as a target"
    assert np.array_equal(mb["attention_mask"], (rows_c != PAD_ID).astype(np.int32)), "attention_mask is wrong"
    assert not (mb["labels"][rows_c == PAD_ID] >= 0).any(), "a PAD position became a target"

    # The optimizer path is otherwise never executed here. A few steps on one
    # fixed batch must drive the loss down; if they do not, gradients are not
    # reaching the parameters and no amount of pretraining will help.
    small = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    small.train()
    opt = make_optimizer(small, trapezoid_schedule(1e-3, 40))
    fixed = to_device(mask_batch(rng.integers(N_SPECIAL, 500, (2, 16)).astype(np.int32), rng, 0.30), None)
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
    args = p.parse_args()

    print(f"smoke test: {args.smoke} ...")

    if args.selftest:
        selftest()
        return

    if args.smoke:
        args.docs, args.seq_len = 200, 128
        args.batch_size, args.layers, args.hidden, args.heads = 8, 4, 128, 4
        args.log_every, args.eval_every, args.save_every = 5, 10, 15
        # leave an explicit --steps alone, so --smoke --resume can be exercised
        if args.steps == p.get_default("steps"):
            args.steps = 30

    print("building corpus ...", flush=True)
    texts = list(iter_texts(args.dataset, args.config, args.docs, args.smoke))

    if args.smoke:
        tokenizer = ByteTokenizer()
        vocab_size = ByteTokenizer.vocab
    else:
        tokenizer = build_tokenizer(texts, args.vocab_size, args.tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

    chunks, doc_ids = chunk_documents(texts, tokenizer, args.seq_len, with_doc_ids=True)
    print(f"corpus: {len(texts)} docs -> {len(chunks)} sequences of {args.seq_len} tokens "
          f"({len(chunks) * args.seq_len / 1e6:.2f}M tokens), vocab {vocab_size}")
    del texts  # gigabytes of Python strings at --docs 200000, and dead from here

    # Hold out whole *documents* before training touches them. Splitting on
    # windows instead would drop two chunks of the same article on opposite
    # sides, and the held-out loss would read back part of the training set.
    n_docs = int(doc_ids.max()) + 1
    doc_perm = np.random.default_rng(args.seed).permutation(n_docs)
    n_val_docs = max(1, int(round(n_docs * args.val_frac)))
    want = args.batch_size * MIN_VAL_BATCHES
    while True:
        is_val = np.isin(doc_ids, doc_perm[:n_val_docs])
        if is_val.sum() >= want or n_val_docs >= n_docs // 2:
            break
        n_val_docs = min(n_val_docs * 2, n_docs // 2)
    val_chunks, train_chunks = chunks[is_val], chunks[~is_val]
    if len(val_chunks) < args.batch_size or len(train_chunks) < args.batch_size:
        raise RuntimeError(
            f"corpus too small to split: {len(chunks)} windows from {n_docs} docs, "
            f"batch {args.batch_size}"
        )
    print(f"split: {len(train_chunks)} train / {len(val_chunks)} held-out windows "
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

    # Offset the stream seed by the resume point so a continued run does not
    # replay the batches it already saw.
    stream = prefetch(batch_stream(train_chunks, args.batch_size, args.mask_prob,
                                   seed=args.seed + 1 + start_step, sharding=data_sharding))

    tokens_per_step = args.batch_size * args.seq_len
    print(f"training steps {start_step + 1}..{args.steps} at {args.mask_prob:.0%} masking, "
          f"{tokens_per_step} tokens/step")

    start = time.time()
    for step in range(start_step + 1, args.steps + 1):
        batch = next(stream)
        loss = train_step(model, optimizer, batch)
        if step % args.log_every == 0 or step == start_step + 1:
            loss = float(loss)
            elapsed = time.time() - start
            done = step - start_step  # this run's steps, not the schedule's
            print(f"step {step:>6}  loss {loss:6.3f}  "
                  f"ppl {math.exp(min(loss, 20)):9.1f}  "
                  f"lr {float(schedule(step - 1)):.2e}  "
                  f"{done * tokens_per_step / max(elapsed, 1e-6) / 1e3:6.1f}k tok/s",
                  flush=True)
        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            val = evaluate(model, val_chunks, args.batch_size, args.mask_prob,
                           sharding=data_sharding)
            model.train()
            print(f"step {step:>6}  held-out loss {val:6.3f}  (train {float(loss):6.3f})",
                  flush=True)
        if mgr is not None and args.save_every and step % args.save_every == 0:
            save_checkpoint(mgr, step, model, optimizer)

    model.eval()
    final = evaluate(model, val_chunks, args.batch_size, args.mask_prob, sharding=data_sharding)
    print(f"final held-out mlm loss {final:.3f}")

    if mgr is not None:
        save_checkpoint(mgr, args.steps, model, optimizer)
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
    rows = val_chunks[:args.batch_size]
    batch = to_device({"ids": rows, "am": (rows != PAD_ID).astype(np.int32)}, data_sharding)
    hidden = model.encode(batch["ids"], batch["am"])
    w = (batch["ids"] >= N_SPECIAL)[..., None].astype(hidden.dtype)
    pooled = (hidden * w).sum(1) / jnp.maximum(w.sum(1), 1.0)
    print(f"pooled embedding shape {pooled.shape}")


if __name__ == "__main__":
    main()