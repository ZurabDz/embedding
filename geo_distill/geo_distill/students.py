"""The two student flavours behind one interface, plus their on-disk config.

train.py and eval.py dispatch through spec_for(...) instead of scattering
`student_type == "mlm"` branches: each spec owns its tokenizer contract, its
row encoding, and how to rebuild the module from a saved config.
"""
from __future__ import annotations

import dataclasses
import json

from mlm import assert_pad_is_zero, assert_special_tokens, encode_sentences

from geo_distill.data import encode_batch


@dataclasses.dataclass
class StudentConfig:
    """Everything eval.py — and a --resume — needs to rebuild the student and
    reproduce its space, without going back to the Hub for the encoder.

    Written next to the params. Loading tolerates unknown keys and fills in
    defaults for fields written before they existed (student_type, dropout), so
    old artifacts keep evaluating.
    """

    student_type: str
    vocab_size: int
    out_dim: int
    max_len: int
    tokenizer: str
    # Whether the regression targets were mean-centered; the training-split
    # mean itself is saved alongside as teacher_mean.npy, so the space the
    # student was trained into stays reproducible after the run exits.
    center: bool = True
    # The dropout the run trained with. A *training* knob in an otherwise
    # eval-oriented struct, because dropout is baked in at construction (the
    # nnx.Dropout rate, and MultiHeadAttention's dropout_rate — which at rate 0
    # forks no rng stream at all): a module rebuilt at 0.0 cannot be trained
    # with dropout, so --resume has to know what to rebuild with. Inference
    # ignores it; rebuild() defaults to 0.0.
    dropout: float = 0.0
    # from-scratch architecture (None for mlm students)
    dim: int | None = None
    depth: int | None = None
    heads: int | None = None
    mlp_dim: int | None = None
    embed_dim: int | None = None
    # pretrained-encoder provenance (None for from-scratch students)
    mlm_checkpoint: str | None = None
    mlm_step: int | None = None
    mlm_encoder: dict | None = None
    # outcome, restamped when training finishes
    best_val_spearman: float | None = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def load(cls, path: str) -> "StudentConfig":
        with open(path) as f:
            d = json.load(f)
        d.setdefault("student_type", "scratch")
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class ScratchSpec:
    """The tiny from-scratch EmbeddingModel."""

    name = "scratch"
    default_lr = 3e-4

    @staticmethod
    def validate_tokenizer(tok) -> None:
        # Rows are padded with 0, so [PAD] must sit there; everything else
        # about the vocabulary is the student's own business — it is trained
        # from scratch on whatever ids the tokenizer emits.
        assert_pad_is_zero(tok)

    @staticmethod
    def encode(tok, sentences, max_len: int):
        return encode_batch(tok, sentences, max_len)

    @staticmethod
    def rebuild(cfg: StudentConfig, *, dropout: float = 0.0, seed: int = 0):
        from flax import nnx

        from geo_distill.model import EmbeddingModel

        model = EmbeddingModel(
            vocab_size=cfg.vocab_size, dim=cfg.dim, depth=cfg.depth,
            heads=cfg.heads, mlp_dim=cfg.mlp_dim, out_dim=cfg.out_dim,
            max_len=cfg.max_len, dropout=dropout, embed_dim=cfg.embed_dim,
            rngs=nnx.Rngs(seed))
        model.eval()  # parity with MlmSpec; train.py flips the mode it wants
        return model


class MlmSpec:
    """The lm-pretrained MLM encoder, fine-tuned (--mlm-checkpoint)."""

    name = "mlm"
    default_lr = 5e-5  # fine-tuning: an order of magnitude gentler than scratch

    @staticmethod
    def validate_tokenizer(tok) -> None:
        # The encoder was pretrained on rows built with CLS_ID=3/SEP_ID=4 and
        # pools everything below N_SPECIAL away. A tokenizer with a different
        # special-token layout (e.g. the old 2-special geo_distill recipe)
        # silently corrupts every row, so this is a hard error, not a warning.
        assert_special_tokens(tok)

    @staticmethod
    def encode(tok, sentences, max_len: int):
        return encode_sentences(tok, sentences, max_len)

    @staticmethod
    def rebuild(cfg: StudentConfig, *, dropout: float = 0.0, seed: int = 0):
        from geo_distill.mlm_student import build_mlm_student_from_config

        return build_mlm_student_from_config(cfg.mlm_encoder, cfg.out_dim,
                                             dropout=dropout, seed=seed)


SPECS = {"scratch": ScratchSpec, "mlm": MlmSpec}


def spec_for(student_type: str):
    try:
        return SPECS[student_type]
    except KeyError:
        raise RuntimeError(f"unknown student_type {student_type!r}; "
                           f"expected one of {sorted(SPECS)}") from None
