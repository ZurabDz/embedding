"""Train a small tokenizer on the Georgian corpus.

A tokenizer trained directly on your data is a big part of what makes a small
model efficient — it spends its limited vocabulary on the subwords that actually
occur in Georgian instead of wasting it on the whole world's scripts.

Two flavours:
  bpe      byte-level BPE — never produces [UNK], but Georgian is 3 bytes per
           character in UTF-8, so tokens are byte fragments.
  unigram  character-level Unigram (NFKC-normalized) — tokens are natural
           character/subword units, usually a better fit for Georgian script.

Output: artifacts/tokenizer.json

Usage:
    python train_tokenizer.py --vocab_size 12000
    python train_tokenizer.py --model unigram --vocab_size 8000
"""
from __future__ import annotations

import argparse
import os

from tokenizers import (Tokenizer, models, trainers, pre_tokenizers, decoders,
                        normalizers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sentences.txt")
    ap.add_argument("--out", default="artifacts/tokenizer.json")
    ap.add_argument("--model", default="bpe", choices=["bpe", "unigram"])
    ap.add_argument("--vocab_size", type=int, default=12000)
    ap.add_argument("--min_frequency", type=int, default=2)
    args = ap.parse_args()

    # [PAD] must be id 0 — encode_batch pads with 0.
    special = ["[PAD]", "[UNK]"]

    if args.model == "bpe":
        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        # Byte-level handles any Unicode (incl. Georgian) with no [UNK] surprises.
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
    else:
        tok = Tokenizer(models.Unigram())
        tok.normalizer = normalizers.NFKC()
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
        trainer = trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=special,
            unk_token="[UNK]",
        )

    tok.train([args.input], trainer)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"Vocab size: {tok.get_vocab_size()}  ->  {args.out}")
    print("[PAD] id:", tok.token_to_id("[PAD]"), " [UNK] id:", tok.token_to_id("[UNK]"))

    demo = "ქართული ენა ლამაზია."
    enc = tok.encode(demo)
    print("Demo:", demo)
    print("  ids   :", enc.ids)
    print("  tokens:", enc.tokens)


if __name__ == "__main__":
    main()
