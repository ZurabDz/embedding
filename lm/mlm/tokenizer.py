"""Tokenizer training, resolution, and the special-token contract.

One recipe for the whole repo: every tokenizer trained here carries
SPECIAL_TOKENS at ids 0..4, which is what the encoder rows, the masking, and
the content-token pooling all hardcode. Grain-free on purpose — downstream
packages (geo_distill) load and validate tokenizers through this module
without the training pipeline installed.
"""

import os

from mlm import hub, progress
from mlm.config import SPECIAL_TOKENS


def assert_special_tokens(tok, *, context: str = "") -> None:
    """Hard check that the five special tokens sit at ids 0..4.

    The pipeline hardcodes that layout everywhere: rows are built with
    CLS_ID=3/SEP_ID=4, masking writes MASK_ID=1, and pooling drops every id
    below N_SPECIAL. A tokenizer with a different layout (e.g. one trained
    with only [PAD],[UNK]) silently corrupts every row, so refuse it loudly.
    """
    expected = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    found = {t: tok.token_to_id(t) for t in SPECIAL_TOKENS}
    if found != expected:
        where = f" ({context})" if context else ""
        raise RuntimeError(
            f"tokenizer{where} does not carry the special tokens this encoder "
            f"requires: expected {expected}, found {found}. Retrain it with "
            f"this repo's recipe (mlm.tokenizer.train_bpe or the geo_distill "
            f"tokenizer command), or use a compatible one such as "
            f"ZurabDz/ka-bpe-32k."
        )


def assert_pad_is_zero(tok) -> None:
    """The minimal contract for encoders that only pad: [PAD] must be id 0,
    because rows are filled with 0 and the pad mask assumes it."""
    pad = tok.token_to_id("[PAD]")
    if pad != 0:
        raise RuntimeError(
            f"tokenizer has [PAD] at id {pad}, expected 0 — rows are padded "
            f"with 0, so any other layout corrupts the pad mask"
        )


def _vocab_note(tokenizer, requested: int) -> None:
    """An authoritative tokenizer (Hub repo or local directory) defines the
    vocabulary; say so out loud when the flag disagrees instead of silently
    ignoring it."""
    n = tokenizer.get_vocab_size()
    if n != requested:
        print(f"  note: --vocab-size {requested} is ignored — "
              f"this tokenizer defines vocab {n}")


def resolve_tokenizer(spec: str, vocab_size: int):
    """A ready tokenizer from `spec`, or None when one must be trained.

    `spec` is tried as, in order: a local directory holding a tokenizer.json
    (e.g. a downloaded Hub repo); an existing local tokenizers-JSON file; an
    hf://user/repo or bare user/repo Hub id (downloaded, never trained over —
    a typo'd repo id errors out rather than silently training into a file of
    that name); anything else returns None and the caller trains a fresh BPE
    and saves it at `spec`. Every tokenizer returned has passed the
    special-token check.
    """
    from tokenizers import Tokenizer

    if os.path.isdir(spec):
        inner = os.path.join(spec, "tokenizer.json")
        if not os.path.isfile(inner):
            raise RuntimeError(
                f"--tokenizer-path {spec} is a directory without a "
                f"tokenizer.json — point at the file itself, or at a Hub repo id"
            )
        tok = Tokenizer.from_file(inner)
        print(f"loaded tokenizer from {inner} (vocab {tok.get_vocab_size()})")
        assert_special_tokens(tok, context=inner)
        _vocab_note(tok, vocab_size)
        return tok
    if os.path.isfile(spec):
        cached = Tokenizer.from_file(spec)
        n = cached.get_vocab_size()
        # A cache smaller than requested is normal — the corpus may not support
        # the full budget. A cache *larger* than requested means --vocab-size was
        # lowered since it was built, and silently reusing it would train a model
        # with a different vocabulary than the flags describe.
        if n <= vocab_size:
            assert_special_tokens(cached, context=spec)
            print(f"reusing tokenizer {spec} (vocab {n})")
            return cached
        print(f"{spec} has vocab {n} > requested {vocab_size}; retraining")
        return None
    if hub.is_hub_id(spec) and not os.path.isdir(os.path.dirname(spec)):
        # A bare user/repo whose parent directory exists locally stays a
        # *training target* (--tokenizer-path tokenizers/kabpe worked before
        # the Hub integration and still does); hf://user/repo always names the
        # Hub, since "hf://user" can never be a local directory.
        tok = hub.load_hub_tokenizer(spec)
        assert_special_tokens(tok, context=spec)
        _vocab_note(tok, vocab_size)
        return tok
    return None


def train_bpe(texts, vocab_size: int, path: str, *, min_frequency: int = 0):
    """Byte-level BPE trained on the target language only, saved to `path`."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    n_docs = f" on {len(texts)} documents" if hasattr(texts, "__len__") else ""
    print(f"training a {vocab_size}-token BPE{n_docs} — "
          f"takes a few minutes on a full corpus ...", flush=True)
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        # Without this, only bytes that occur in the training corpus enter the
        # vocabulary and everything else falls back to [UNK] at inference — for
        # a Georgian corpus that silently guts Latin, digits and punctuation.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        # The Rust bar redraws with carriage returns, which only a real terminal
        # renders; piped or notebook-captured output would get one line
        # kilometres wide.
        show_progress=progress.raw_terminal(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(path)
    assert_special_tokens(tok, context=path)
    return tok
