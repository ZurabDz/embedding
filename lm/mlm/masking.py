"""Dynamic MLM masking: 30% of maskable positions -> [MASK], no 80/10/10."""

import grain
import numpy as np

from mlm.config import MASK_ID, N_SPECIAL, PAD_ID


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
