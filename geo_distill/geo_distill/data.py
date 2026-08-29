"""Corpus download and the text -> arrays helpers every stage shares:
tokenizer loading, batch encoding for the from-scratch student, and the
stable content-based train/val split.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

import numpy as np

# Sentence enders: Latin-style punctuation (used in modern Georgian) + newlines.
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
# A string counts as "Georgian" if it contains Mkhedruli characters.
_HAS_GEORGIAN = re.compile(r"[\u10D0-\u10FF]")
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Tokenizer loading + encoding
# --------------------------------------------------------------------------- #
def load_tokenizer(spec: str):
    """A tokenizers-JSON file path, or a Hub repo id (e.g. ZurabDz/ka-bpe-32k).

    Hub ids resolve through mlm.hub, which also gets authentication right for
    private repos: the Colab secrets vault, HF_TOKEN, and the token file
    written by `hf auth login` are all honoured.
    """
    from tokenizers import Tokenizer

    if os.path.isfile(spec):
        return Tokenizer.from_file(spec)
    from mlm import hub

    if hub.is_hub_id(spec):
        return hub.load_hub_tokenizer(spec)
    raise FileNotFoundError(f"tokenizer not found: {spec}")


def encode_batch(tokenizer, sentences, max_len: int, pad_id: int = 0):
    """Turn a list of strings into (tokens, pad_mask) numpy arrays.

    tokens:   (N, max_len) int32, right-padded with pad_id
    pad_mask: (N, max_len) float32, 1 for real tokens and 0 for padding
    """
    n = len(sentences)
    tokens = np.full((n, max_len), pad_id, dtype=np.int32)
    mask = np.zeros((n, max_len), dtype=np.float32)
    for i, enc in enumerate(tokenizer.encode_batch(list(sentences))):
        ids = enc.ids[:max_len]
        tokens[i, : len(ids)] = ids
        mask[i, : len(ids)] = 1.0
    return tokens, mask


# --------------------------------------------------------------------------- #
# Train / val split
# --------------------------------------------------------------------------- #
def val_split(sentences, val_frac: float, seed: int = 0):
    """Deterministic, content-based train/val split.

    A sentence lands in val or train based only on a hash of its *text* (plus the
    seed) — never on the dataset's size or order. So as you embed more data, every
    already-seen sentence keeps its assignment and the validation set stays stable,
    which makes metrics comparable across runs with different amounts of data.
    (The old random-permutation split re-drew a different val set every run, so the
    score moved as much because of the changing benchmark as the model.)

    Returns (train_idx, val_idx) as int arrays.
    """
    thr = int(val_frac * 1_000_000)
    train_idx, val_idx = [], []
    for i, s in enumerate(sentences):
        h = int(hashlib.sha1(f"{seed}\x00{s}".encode("utf-8")).hexdigest(), 16)
        (val_idx if (h % 1_000_000) < thr else train_idx).append(i)
    # Guard tiny corpora so neither split is ever empty.
    if len(val_idx) < 2 or not train_idx:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(sentences))
        n_val = max(2, int(len(sentences) * val_frac))
        val_idx, train_idx = perm[:n_val].tolist(), perm[n_val:].tolist()
    return np.array(train_idx), np.array(val_idx)


def metric_subset(val_sentences, max_rows: int, seed: int = 0):
    """Positions inside the val split that the O(n^2) agreement metric scores.

    None when the whole split already fits. The metric builds two dense n x n
    similarity matrices in host RAM, so its cost is quadratic: 50k held-out
    sentences ask for 20 GB *each*, which arrives as an OOM kill at the first
    epoch boundary rather than as an error.

    Chosen by content hash rather than by position, so the subset depends only
    on *which* sentences are held out: the same set every epoch, in a rerun and
    across a --resume, which is what makes the epoch-to-epoch scores and the
    best-checkpoint comparison mean anything. A separate salt keeps the draw
    independent of the one that put the sentence in val to begin with.

    Unlike val_split this is a fixed *size*, not a fixed hash threshold —
    memory is the binding constraint here, and a threshold's subset grows with
    the corpus. So growing the corpus re-draws this subset (overlap decays as
    the ratio of the two val sizes): scores are comparable within a corpus, not
    across two different ones.
    """
    n = len(val_sentences)
    if not max_rows or n <= max_rows:
        return None
    # First 64 bits of the digest: enough to order 50k sentences without ties,
    # and an int64 argsort instead of a Python-object one.
    h = np.array([int(hashlib.sha1(f"{seed}\x01{s}".encode("utf-8")).hexdigest()[:16], 16)
                  for s in val_sentences], dtype=np.uint64)
    return np.sort(np.argsort(h, kind="stable")[:max_rows])


# --------------------------------------------------------------------------- #
# Corpus download (the `data` subcommand)
# --------------------------------------------------------------------------- #
def pick_text_column(features) -> str:
    names = list(features.keys())
    for pref in ("text", "content", "sentence", "body", "raw"):
        if pref in names:
            return pref
    # else: first column whose values are strings
    for name in names:
        if getattr(features[name], "dtype", None) == "string":
            return name
    return names[0]


def to_sentences(text: str, min_chars: int, max_chars: int):
    for chunk in _SENT_SPLIT.split(text or ""):
        s = _WS.sub(" ", chunk).strip()
        if min_chars <= len(s) <= max_chars and _HAS_GEORGIAN.search(s):
            yield s


def run(args) -> None:
    """Stream a HF dataset and write clean, deduplicated Georgian sentences."""
    from datasets import load_dataset

    # Stream so we don't download the whole thing when we only need a slice.
    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    text_col = pick_text_column(ds.features) if ds.features else None
    print(f"Using text column: {text_col!r}")

    seen = set()
    out = []
    for row in ds:
        text = row[text_col] if text_col else next(iter(row.values()))
        for s in to_sentences(text, args.min_chars, args.max_chars):
            if s not in seen:
                seen.add(s)
                out.append(s)
        if len(out) >= args.n:
            break

    out = out[: args.n]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Wrote {len(out)} sentences -> {args.out}")
    if out:
        print("Sample:", out[0][:80])

    # `datasets` streaming pulls data over fsspec/aiohttp, which runs an asyncio
    # event loop in a daemon thread. That native thread can be torn down mid-flight
    # during interpreter shutdown, aborting with:
    #   Fatal Python error: PyGILState_Release ... must be current when releasing
    # The file is already written and flushed above, so exit now — before the
    # buggy finalization runs — to guarantee a clean exit code.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
