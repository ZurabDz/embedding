"""Student built on the pretrained Georgian MLM encoder from ../lm.

Instead of learning Georgian *and* the teacher's geometry from scratch, this
student starts from the MLM-pretrained encoder (e.g. hf.co/ZurabDz/ka-mlm) and
its 32k BPE (hf.co/ZurabDz/ka-bpe-32k): a projection head maps the pooled
hidden state into the teacher's embedding space, and the whole stack is
fine-tuned by the same distillation loop as the from-scratch student.

The encoder lives in the sibling `lm/` project (the `mlm` package); this module
puts that directory on sys.path, so cloning the whole `embedding` repo is the
only setup — locally and on Kaggle/Colab alike.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_LM_DIR = Path(__file__).resolve().parent.parent / "lm"
if str(_LM_DIR) not in sys.path:
    sys.path.insert(0, str(_LM_DIR))

import jax.numpy as jnp
import numpy as np
from flax import nnx

from mlm.config import CLS_ID, N_SPECIAL, PAD_ID, SEP_ID, EncoderConfig
from mlm.model import MlmEncoder

from model import l2_normalize


def encode_batch_mlm(tokenizer, sentences, max_len: int):
    """Sentences -> ([CLS] ids [SEP] + padding, mask) — the exact row shape the
    encoder was pretrained on. Returns the same (tokens, pad_mask) pair as
    distill_lib.encode_batch, so the training loop is student-agnostic."""
    n = len(sentences)
    tokens = np.full((n, max_len), PAD_ID, dtype=np.int32)
    mask = np.zeros((n, max_len), dtype=np.float32)
    body = max_len - 2  # room for [CLS] and [SEP]
    for i, enc in enumerate(tokenizer.encode_batch(list(sentences))):
        ids = enc.ids[:body]
        row = [CLS_ID, *ids, SEP_ID]
        tokens[i, : len(row)] = row
        mask[i, : len(row)] = 1.0
    return tokens, mask


class MlmStudent(nnx.Module):
    """Pretrained encoder -> content-token mean-pool -> projection -> L2 norm.

    Pooling drops [CLS]/[SEP]/[PAD] (everything below N_SPECIAL): [CLS] is an
    untrained attention sink in MLM pretraining, and averaging it in would fold
    a high-norm outlier into every sentence embedding.
    """

    def __init__(self, encoder: MlmEncoder, out_dim: int, *, rngs: nnx.Rngs):
        self.encoder = encoder
        self.proj = nnx.Linear(
            encoder.cfg.hidden, out_dim,
            kernel_init=nnx.initializers.normal(0.02), rngs=rngs,
        )

    def __call__(self, tokens, pad_mask, deterministic=True):
        # `deterministic` exists for EmbeddingModel API parity; the encoder's
        # dropout follows train()/eval() mode, which train.py sets around the
        # loop and its eval passes.
        hidden = self.encoder.encode(tokens, pad_mask.astype(jnp.int32))
        w = (tokens >= N_SPECIAL)[..., None].astype(hidden.dtype)
        pooled = (hidden * w).sum(axis=1) / jnp.maximum(w.sum(axis=1), 1.0)
        return l2_normalize(self.proj(pooled))


def encoder_config_to_dict(cfg: EncoderConfig) -> dict:
    """JSON-safe encoder config, so eval.py can rebuild the architecture from
    student_config.json alone — no Hub access or checkpoint re-download."""
    import dataclasses

    d = dataclasses.asdict(cfg)
    for k in ("dtype", "param_dtype"):
        d[k] = getattr(cfg, k).__name__
    return d


def encoder_config_from_dict(d: dict) -> EncoderConfig:
    d = dict(d)
    for k in ("dtype", "param_dtype"):
        if k in d:
            d[k] = getattr(jnp, d[k])
    return EncoderConfig(**d)


def load_mlm_student(spec: str, out_dim: int, *, dropout: float = 0.0,
                     seed: int = 0, expect_vocab: int | None = None,
                     cache_dir: str = "artifacts/mlm_checkpoint"):
    """Build an MlmStudent from a pretrained checkpoint.

    `spec` is a local lm --save-dir (holding config.json + <step>/) or a Hub
    repo id pushed by lm's --hub-checkpoints (e.g. ZurabDz/ka-mlm) — a private
    repo needs HF_TOKEN / `hf auth login`. Returns (student, encoder_cfg, step).
    """
    from mlm import hub
    from mlm.checkpoint import load_checkpoint

    if os.path.isdir(spec) and os.path.isfile(os.path.join(spec, "config.json")):
        ckpt_dir = spec
    else:
        # Cache keyed by repo id: pull_checkpoint's "local wins ties" contract
        # would otherwise silently serve repo A's cached checkpoint to a run
        # that asked for repo B.
        ckpt_dir = os.path.join(cache_dir,
                                spec.removeprefix("hf://").replace("/", "__"))
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            # crash-safe pull; a warm cache that already holds the step is a no-op
            hub.pull_checkpoint(ckpt_dir, spec)
        except RuntimeError as e:
            # lm's "first session / --resume" wording is meaningless here, and
            # a private repo without a token looks identical to a typo'd id.
            raise RuntimeError(
                f"could not fetch a checkpoint from hf.co/{spec}: the id may "
                f"be mistyped, the repo may be private (set HF_TOKEN or run "
                f"`hf auth login`), or it holds no checkpoint yet. "
                f"--mlm_checkpoint also accepts a local lm --save-dir."
            ) from e
    encoder, cfg, step = load_checkpoint(ckpt_dir, dropout=dropout)
    if expect_vocab is not None and cfg.vocab_size != expect_vocab:
        raise RuntimeError(
            f"tokenizer vocab {expect_vocab} != encoder vocab {cfg.vocab_size}: "
            f"the student must use the SAME tokenizer the encoder was "
            f"pretrained with (for ZurabDz/ka-mlm that is ZurabDz/ka-bpe-32k)"
        )
    print(f"loaded MLM encoder from {spec} (step {step}, hidden {cfg.hidden}, "
          f"layers {cfg.layers}, vocab {cfg.vocab_size}, dropout {cfg.dropout})")
    return MlmStudent(encoder, out_dim, rngs=nnx.Rngs(seed)), cfg, step


def build_mlm_student_from_config(student_cfg: dict) -> MlmStudent:
    """Rebuild the architecture from student_config.json (eval path); the
    fine-tuned weights come from student_params.msgpack afterwards."""
    enc = MlmEncoder(encoder_config_from_dict(student_cfg["mlm_encoder"]),
                     rngs=nnx.Rngs(0))
    model = MlmStudent(enc, student_cfg["out_dim"], rngs=nnx.Rngs(0))
    model.eval()
    return model
