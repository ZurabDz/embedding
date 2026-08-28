"""The contracts other packages (and future selves) rely on from `mlm`:
a lean import surface, a stable config codec, and the checkpoint round-trip."""
import json
import subprocess
import sys

import numpy as np


def test_import_mlm_is_lean():
    """`import mlm` must not pull in grain/orbax/datasets — geo_distill and any
    fine-tuning env import it without the training pipeline installed."""
    code = (
        "import sys; import mlm; "
        "heavy = [m for m in sys.modules if m == 'grain' "
        "or m.startswith(('grain.', 'orbax', 'datasets'))]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_teacher_is_jax_free():
    """The teacher stage (Gemini client + synthetic stand-in) deliberately
    avoids jax so it stays cheap to run anywhere."""
    code = (
        "import sys; import geo_distill.teacher; "
        "assert 'jax' not in sys.modules and 'flax' not in sys.modules, "
        "[m for m in sys.modules if m in ('jax', 'flax')]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_import_local_teacher_is_torch_free():
    """The local-teacher stage keeps torch/transformers (and jax) out of its
    import chain — they load lazily inside make_encoder, so the CLI and the
    offline tests never need them installed."""
    code = (
        "import sys; import geo_distill.local_teacher, geo_distill.hub; "
        "heavy = [m for m in sys.modules if m in ('torch', 'jax', 'flax') "
        "or m.startswith('transformers')]; "
        "assert not heavy, heavy"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_legacy_config_loads():
    """The exact shape of a pre-param_dtype config.json (what old local runs
    and the ka-mlm Hub repo hold) must keep loading, with fp32 defaults."""
    from mlm.config import EncoderConfig

    legacy = {"vocab_size": 261, "hidden": 128, "layers": 4, "heads": 4,
              "mlp_hidden": 320, "max_len": 128, "rope_theta": 10000.0,
              "norm_eps": 1e-6, "dropout": 0.0, "init_std": 0.02,
              "dtype": "float32"}  # no param_dtype, no remat
    cfg = EncoderConfig.from_json_dict(legacy)
    assert cfg.param_dtype.__name__ == "float32"
    assert cfg.remat is False
    # and the codec round-trips losslessly
    assert EncoderConfig.from_json_dict(cfg.to_json_dict()) == cfg


def test_checkpoint_roundtrip_and_layout(tiny_checkpoint):
    from mlm.checkpoint import load_checkpoint

    d, cfg = tiny_checkpoint
    # on-disk layout is the contract hub.push_checkpoint mirrors
    assert (d / "config.json").is_file()
    assert (d / "1" / "model").is_dir()
    assert (d / "1" / "opt").is_dir()
    assert json.loads((d / "config.json").read_text())["vocab_size"] == cfg.vocab_size

    m1, c1, step = load_checkpoint(str(d))
    m2, _, _ = load_checkpoint(str(d))
    assert step == 1 and c1 == cfg
    ids = np.array([[3, 10, 11, 12, 4, 0, 0, 0]], dtype=np.int32)
    am = (ids != 0).astype(np.int32)
    h1 = np.asarray(m1.encode(ids, am))
    h2 = np.asarray(m2.encode(ids, am))
    np.testing.assert_array_equal(h1, h2)

    # the dropout override changes only the config, never the weights
    m3, c3, _ = load_checkpoint(str(d), dropout=0.2)
    assert c3.dropout == 0.2
    m3.eval()
    np.testing.assert_array_equal(np.asarray(m3.encode(ids, am)), h1)


def test_cls_row_layout():
    from mlm.config import CLS_ID, PAD_ID, SEP_ID
    from mlm.encoding import cls_row

    row = cls_row([10, 11, 12], 8)
    assert row.tolist() == [CLS_ID, 10, 11, 12, SEP_ID, PAD_ID, PAD_ID, PAD_ID]
    assert row.dtype == np.int32


def test_windows_use_cls_row(bpe_tokenizer):
    """The pretraining windower and cls_row must agree exactly — the
    train/val ArrayRecords are built from _windows."""
    from mlm.config import MIN_WINDOW_TOKENS
    from mlm.data import _windows
    from mlm.encoding import cls_row

    ids = list(range(10, 10 + 100))
    seq_len = 48
    body = seq_len - 2
    rows = list(_windows(ids, seq_len))
    expected = [cls_row(ids[start:start + body], seq_len)
                for start in range(0, len(ids), body)
                if len(ids[start:start + body]) >= MIN_WINDOW_TOKENS]
    assert len(rows) == len(expected) > 0
    for got, want in zip(rows, expected):
        np.testing.assert_array_equal(got, want)
