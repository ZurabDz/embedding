"""Save -> load -> identical embeddings, for both student types, plus the
StudentConfig compatibility guarantees."""
import json

import numpy as np
import pytest

from geo_distill.checkpoint import (load_student, save_student_config,
                                    save_student_params)
from geo_distill.model import embed_in_batches
from geo_distill.students import StudentConfig, spec_for


def _roundtrip(tmp_path, model, cfg, tok, sentences):
    save_student_config(str(tmp_path), cfg)
    save_student_params(str(tmp_path), model)
    spec = spec_for(cfg.student_type)
    tokens, mask = spec.encode(tok, sentences[:8], cfg.max_len)
    model.eval()
    before = embed_in_batches(model, tokens, mask)

    loaded, loaded_cfg, loaded_spec = load_student(str(tmp_path))
    after = embed_in_batches(loaded, tokens, mask)
    np.testing.assert_array_equal(before, after)
    assert loaded_cfg.student_type == cfg.student_type
    assert loaded_spec is spec
    # embeddings come out unit-norm from both students
    np.testing.assert_allclose(np.linalg.norm(after, axis=-1), 1.0, atol=1e-5)


def test_scratch_roundtrip(tmp_path, bpe_tokenizer, sentences):
    from flax import nnx

    from geo_distill.model import EmbeddingModel

    cfg = StudentConfig(student_type="scratch",
                        vocab_size=bpe_tokenizer.get_vocab_size(), out_dim=16,
                        max_len=16, tokenizer="unused.json", center=True,
                        dim=32, depth=1, heads=2, mlp_dim=64, embed_dim=None)
    model = EmbeddingModel(vocab_size=cfg.vocab_size, dim=32, depth=1, heads=2,
                           mlp_dim=64, max_len=16, out_dim=16, dropout=0.0,
                           rngs=nnx.Rngs(0))
    _roundtrip(tmp_path, model, cfg, bpe_tokenizer, sentences)


def test_mlm_roundtrip_and_head_strip(tmp_path, tiny_checkpoint, bpe_tokenizer,
                                      sentences):
    from flax import nnx

    from geo_distill.mlm_student import load_mlm_student

    ck_dir, enc_cfg = tiny_checkpoint
    model, cfg_loaded, step = load_mlm_student(str(ck_dir), out_dim=16)
    assert cfg_loaded == enc_cfg and step == 1

    # the MLM head is dead weight for embedding and must not be in the tree
    paths = {"/".join(str(p) for p in path)
             for path, _ in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    assert not any("head" in p for p in paths), paths
    # ...but the tied token table must stay — it is the input embedding
    assert any("tok" in p for p in paths)

    cfg = StudentConfig(student_type="mlm", vocab_size=enc_cfg.vocab_size,
                        out_dim=16, max_len=32, tokenizer="unused.json",
                        center=False, mlm_checkpoint=str(ck_dir), mlm_step=step,
                        mlm_encoder=enc_cfg.to_json_dict())
    _roundtrip(tmp_path, model, cfg, bpe_tokenizer, sentences)


def test_student_config_tolerates_legacy_and_unknown_keys(tmp_path):
    legacy = {  # a pre-refactor scratch config: no student_type/center, extra keys
        "vocab_size": 500, "dim": 32, "depth": 1, "heads": 2, "mlp_dim": 64,
        "out_dim": 16, "max_len": 16, "dropout": 0.0, "embed_dim": None,
        "tokenizer": "artifacts/tokenizer.json", "best_val_spearman": 0.5,
        "some_future_field": {"nested": True},
    }
    p = tmp_path / "student_config.json"
    p.write_text(json.dumps(legacy))
    cfg = StudentConfig.load(str(p))
    assert cfg.student_type == "scratch"
    assert cfg.center is True  # centering was always the old default
    assert cfg.dim == 32 and cfg.best_val_spearman == 0.5


def test_unknown_student_type_errors(tmp_path):
    p = tmp_path / "student_config.json"
    p.write_text(json.dumps({"student_type": "bogus", "vocab_size": 1,
                             "out_dim": 1, "max_len": 1, "tokenizer": "x"}))
    with pytest.raises(RuntimeError, match="bogus"):
        spec_for(StudentConfig.load(str(p)).student_type)
