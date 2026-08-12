"""Download Georgian text from HuggingFace and turn it into clean sentences.

Default source: ZurabDz/geo_small_corpus. You can use any dataset with a text column.

Output: data/sentences.txt  (one sentence per line, deduplicated, length-filtered)

Usage:
    python data.py --n 20000
    python data.py --dataset <corpus_name> --n 50000
"""
from __future__ import annotations

import argparse
import os
import re
import sys

# Sentence enders: Latin-style punctuation (used in modern Georgian) + newlines.
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
# A string counts as "Georgian" if it contains Mkhedruli characters.
_HAS_GEORGIAN = re.compile(r"[\u10D0-\u10FF]")
_WS = re.compile(r"\s+")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ZurabDz/geo_small_corpus")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20000, help="target number of sentences")
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--max_chars", type=int, default=300)
    ap.add_argument("--out", default="data/sentences.txt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so --help works without it

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


if __name__ == "__main__":
    main()
