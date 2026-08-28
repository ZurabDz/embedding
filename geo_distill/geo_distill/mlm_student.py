"""Student built on the pretrained Georgian MLM encoder (the `mlm` package
from the sibling lm/ project, installed as the ka-mlm workspace member).

Instead of learning Georgian *and* the teacher's geometry from scratch, this
student starts from the MLM-pretrained encoder (e.g. hf.co/ZurabDz/ka-mlm) and
its 32k BPE (hf.co/ZurabDz/ka-bpe-32k): a projection head maps the pooled
hidden state into the teacher's embedding space, and the whole stack is
fine-tuned by the same distillation loop as the from-scratch student.
"""
from __future__ import annotations

import os

import jax.numpy as jnp
from flax import nnx

from mlm import EncoderConfig, MlmEncoder, pool_content
from mlm.checkpoint import load_checkpoint

from geo_distill import config as paths
from geo_distill.model import l2_normalize


class MlmStudent(nnx.Module):
    """Pretrained encoder -> content-token mean-pool -> projection -> L2 norm.

    The encoder's MLM head is dead weight here — distillation only ever calls
    .encode() — so it is dropped from the module tree: its parameters would
    otherwise be optimized and serialized for nothing. Consequence: the wrapped
    encoder can no longer produce logits (encoder(...) would crash); students
    only embed. The tied token table stays — it *is* the input embedding.
    """

    def __init__(self, encoder: MlmEncoder, out_dim: int, *, rngs: nnx.Rngs):
        encoder.head = None  # see docstring
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
        return l2_normalize(self.proj(pool_content(hidden, tokens)))


def load_mlm_student(spec: str, out_dim: int, *, dropout: float = 0.0,
                     seed: int = 0, expect_vocab: int | None = None,
                     cache_dir: str = paths.MLM_CKPT_CACHE):
    """Build an MlmStudent from a pretrained checkpoint.

    `spec` is a local lm --save-dir (holding config.json + <step>/) or a Hub
    repo id pushed by lm's --hub-checkpoints (e.g. ZurabDz/ka-mlm) — a private
    repo needs HF_TOKEN / `hf auth login`. Returns (student, encoder_cfg, step).
    """
    from mlm import hub

    if os.path.isdir(spec) and os.path.isfile(os.path.join(spec, "config.json")):
        ckpt_dir = spec
    else:
        # Cache keyed by repo id: pull_checkpoint's "local wins ties" contract
        # would otherwise silently serve repo A's cached checkpoint to a run
        # that asked for repo B.
        ckpt_dir = os.path.join(cache_dir,
                                hub.strip_prefix(spec).replace("/", "__"))
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            # crash-safe pull; a warm cache that already holds the step is a no-op
            hub.pull_checkpoint(ckpt_dir, spec)
        except Exception as e:
            # Everything the Hub client can throw lands here (HfHubHTTPError,
            # validation errors, lm's own RuntimeError): lm's "first session /
            # --resume" wording is meaningless in this context, and a private
            # repo without a token looks identical to a typo'd id.
            raise RuntimeError(
                f"could not fetch a checkpoint from hf.co/{spec}: the id may "
                f"be mistyped, the repo may be private (set HF_TOKEN or run "
                f"`hf auth login`), or it holds no checkpoint yet. "
                f"--mlm-checkpoint also accepts a local lm --save-dir."
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


def build_mlm_student_from_config(mlm_encoder: dict, out_dim: int) -> MlmStudent:
    """Rebuild the architecture from student_config.json's mlm_encoder dict
    (eval path — no Hub access); the fine-tuned weights come from
    student_params.msgpack afterwards."""
    enc = MlmEncoder(EncoderConfig.from_json_dict(mlm_encoder), rngs=nnx.Rngs(0))
    model = MlmStudent(enc, out_dim, rngs=nnx.Rngs(0))
    model.eval()
    return model
