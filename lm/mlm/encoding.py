"""The encoder's wire format and pooling, shared with downstream consumers.

Grain-free on purpose: fine-tuning packages (geo_distill) import these without
the training pipeline installed. Everything the pipeline hardcodes about rows —
[CLS] at position 0, [SEP] after the body, [PAD] fill, content = ids >=
N_SPECIAL — lives here, so pretraining and downstream encoders cannot drift.
"""

import jax.numpy as jnp
import numpy as np

from mlm.config import CLS_ID, N_SPECIAL, PAD_ID, SEP_ID


def cls_row(ids, seq_len: int) -> np.ndarray:
    """Token ids -> one padded [CLS] ids [SEP] row of exactly seq_len.

    The single definition of the row layout; the pretraining windower and the
    sentence encoder below both assemble rows here. `ids` must already fit:
    len(ids) <= seq_len - 2.
    """
    row = np.full(seq_len, PAD_ID, dtype=np.int32)
    row[0] = CLS_ID
    row[1:1 + len(ids)] = ids
    row[1 + len(ids)] = SEP_ID
    return row


def encode_sentences(tokenizer, sentences, max_len: int):
    """Sentences -> ([CLS] ids [SEP] + padding, pad mask), one row per sentence.

    The exact row shape the encoder was pretrained on. Sentences longer than
    max_len - 2 tokens are truncated — pretraining windows long documents into
    several rows instead, but a sentence encoder wants exactly one row per
    input, never zero or two.
    """
    n = len(sentences)
    tokens = np.full((n, max_len), PAD_ID, dtype=np.int32)
    mask = np.zeros((n, max_len), dtype=np.float32)
    body = max_len - 2  # room for [CLS] and [SEP]
    for i, enc in enumerate(tokenizer.encode_batch(list(sentences))):
        ids = enc.ids[:body]
        tokens[i] = cls_row(ids, max_len)
        mask[i, : len(ids) + 2] = 1.0
    return tokens, mask


def pool_content(hidden, token_ids):
    """Mean-pool hidden states over content tokens only.

    `>= N_SPECIAL` drops [PAD], [CLS], [SEP] and [MASK] at once — [CLS] in
    particular is an untrained attention sink under MLM pretraining, so
    averaging it in would fold a high-norm outlier into every embedding. The
    count is clamped so an all-special row divides by 1 instead of 0.
    """
    w = (token_ids >= N_SPECIAL)[..., None].astype(hidden.dtype)
    return (hidden * w).sum(axis=1) / jnp.maximum(w.sum(axis=1), 1.0)
