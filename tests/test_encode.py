"""Row layouts and the tokenizer contracts — the wire format both students
and the pretrained encoder must agree on."""
import numpy as np
import pytest

from mlm.config import CLS_ID, PAD_ID, SEP_ID
from mlm.encoding import encode_sentences
from mlm.tokenizer import assert_pad_is_zero, assert_special_tokens

from geo_distill.data import encode_batch


def test_encode_sentences_layout(bpe_tokenizer, sentences):
    max_len = 16
    tokens, mask = encode_sentences(bpe_tokenizer, sentences[:8], max_len)
    assert tokens.shape == (8, max_len) and tokens.dtype == np.int32
    assert mask.shape == (8, max_len) and mask.dtype == np.float32
    for row, m in zip(tokens, mask):
        assert row[0] == CLS_ID
        n = int(m.sum())
        assert n >= 3                       # CLS + at least one token + SEP
        assert row[n - 1] == SEP_ID
        assert (row[n:] == PAD_ID).all()    # PAD-filled tail
        assert (m[:n] == 1.0).all() and (m[n:] == 0.0).all()


def test_encode_sentences_truncates_not_drops(bpe_tokenizer):
    long = " ".join(["ქართული ენა ლამაზია"] * 50)
    tokens, mask = encode_sentences(bpe_tokenizer, [long], 12)
    assert tokens.shape == (1, 12)
    assert tokens[0, 0] == CLS_ID and tokens[0, 11] == SEP_ID  # body = max_len - 2
    assert mask[0].sum() == 12


def test_encode_batch_layout(bpe_tokenizer, sentences):
    max_len = 16
    tokens, mask = encode_batch(bpe_tokenizer, sentences[:8], max_len)
    assert tokens.shape == (8, max_len) and tokens.dtype == np.int32
    for sent, row, m in zip(sentences[:8], tokens, mask):
        n = int(m.sum())
        assert n >= 1
        assert (row[n:] == 0).all()         # right-padded with pad_id 0
        # no CLS/SEP framing in this format: the row IS the raw encoding
        assert row[:n].tolist() == bpe_tokenizer.encode(sent).ids[:max_len]


def test_encode_batch_truncates(bpe_tokenizer):
    long = " ".join(["ქართული ენა ლამაზია"] * 50)
    tokens, mask = encode_batch(bpe_tokenizer, [long], 12)
    assert tokens.shape == (1, 12) and mask[0].sum() == 12


def test_special_token_guard(bpe_tokenizer, two_special_tokenizer):
    assert_special_tokens(bpe_tokenizer)  # the shared recipe passes

    with pytest.raises(RuntimeError) as e:
        assert_special_tokens(two_special_tokenizer)
    # the message must show what was found and name a fix
    assert "[CLS]" in str(e.value) and "ka-bpe-32k" in str(e.value)


def test_pad_is_zero_guard(bpe_tokenizer, two_special_tokenizer, sentences):
    assert_pad_is_zero(bpe_tokenizer)
    assert_pad_is_zero(two_special_tokenizer)  # old recipe still fine for scratch

    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    bad = Tokenizer(models.BPE(unk_token="[UNK]"))
    bad.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=400, min_frequency=1, special_tokens=["[UNK]", "[PAD]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    bad.train_from_iterator(sentences, trainer)
    with pytest.raises(RuntimeError):
        assert_pad_is_zero(bad)  # [PAD] landed at id 1
