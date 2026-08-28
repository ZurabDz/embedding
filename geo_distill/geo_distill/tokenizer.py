"""Train a small tokenizer on the Georgian corpus.

A tokenizer trained directly on your data is a big part of what makes a small
model efficient — it spends its limited vocabulary on the subwords that actually
occur in Georgian instead of wasting it on the whole world's scripts.

Two flavours:
  bpe      byte-level BPE — never produces [UNK], but Georgian is 3 bytes per
           character in UTF-8, so tokens are byte fragments. Delegates to the
           shared mlm recipe, so it is the exact trainer lm pretraining uses.
  unigram  character-level Unigram (NFKC-normalized) — tokens are natural
           character/subword units, usually a better fit for Georgian script.

Both flavours reserve mlm's five special tokens at ids 0..4, so any tokenizer
trained here also satisfies the MLM student's contract (--mlm-checkpoint).
"""
from __future__ import annotations

import os

from mlm import SPECIAL_TOKENS, assert_special_tokens
from mlm.tokenizer import train_bpe


def run(args) -> None:
    with open(args.input, encoding="utf-8") as f:
        texts = [ln.strip() for ln in f if ln.strip()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.model == "bpe":
        tok = train_bpe(texts, args.vocab_size, args.out,
                        min_frequency=args.min_frequency)
    else:
        from tokenizers import (Tokenizer, decoders, models, normalizers,
                                pre_tokenizers, trainers)

        tok = Tokenizer(models.Unigram())
        tok.normalizer = normalizers.NFKC()
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
        trainer = trainers.UnigramTrainer(
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS,  # same five, same order, ids 0..4
            unk_token="[UNK]",
        )
        tok.train_from_iterator(texts, trainer)
        tok.save(args.out)
        assert_special_tokens(tok, context=args.out)

    print(f"Vocab size: {tok.get_vocab_size()}  ->  {args.out}")
    print("special ids:", {t: tok.token_to_id(t) for t in SPECIAL_TOKENS})

    demo = "ქართული ენა ლამაზია."
    enc = tok.encode(demo)
    print("Demo:", demo)
    print("  ids   :", enc.ids)
    print("  tokens:", enc.tokens)
