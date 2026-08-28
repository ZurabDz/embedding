"""Shared fixtures. Everything is built through the packages' own public
functions (train_bpe, save_config, save_checkpoint), never by hand-writing
files, so the fixtures cannot drift from what real runs produce. Models are
kept tiny (2 layers / 64 hidden) so the whole suite stays fast on a CPU."""
import pathlib

import pytest

SAMPLES = pathlib.Path(__file__).parent / "data" / "sample_sentences.txt"


@pytest.fixture(scope="session")
def sentences():
    return [ln.strip() for ln in SAMPLES.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


@pytest.fixture(scope="session")
def bpe_tokenizer(tmp_path_factory, sentences):
    """A real 5-special BPE trained by the shared recipe."""
    from mlm.tokenizer import train_bpe

    path = tmp_path_factory.mktemp("tok") / "tok.json"
    return train_bpe(sentences, 600, str(path), min_frequency=1)


@pytest.fixture(scope="session")
def two_special_tokenizer(sentences):
    """The OLD geo_distill recipe (only [PAD],[UNK] reserved) — the reject case
    for the special-token guard."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, min_frequency=1, special_tokens=["[PAD]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(sentences, trainer)
    return tok


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory, bpe_tokenizer):
    """A real orbax checkpoint directory (config.json + step 1) for a tiny
    encoder whose vocab matches bpe_tokenizer. Returns (dir, cfg)."""
    from flax import nnx
    from mlm.checkpoint import checkpoint_manager, save_checkpoint, save_config
    from mlm.config import EncoderConfig
    from mlm.model import MlmEncoder
    from mlm.optim import make_optimizer, trapezoid_schedule

    d = tmp_path_factory.mktemp("ck")
    cfg = EncoderConfig(vocab_size=bpe_tokenizer.get_vocab_size(), hidden=64,
                        layers=2, heads=4, mlp_hidden=128, max_len=64)
    model = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    mgr = checkpoint_manager(str(d))  # creates the dir, same order as lm's train
    save_config(str(d), cfg)
    save_checkpoint(mgr, 1, model,
                    make_optimizer(model, trapezoid_schedule(1e-3, 10)))
    mgr.wait_until_finished()
    mgr.close()
    return d, cfg
