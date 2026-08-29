"""--backend vllm, offline: dispatch, the encode() output contract, cache-key
separation, and run()'s whole-shard handoff — all against a fake `vllm`
module in sys.modules, so the suite never needs vLLM, a GPU, or a download.
(vLLM is deliberately not installable here: it pins its own torch.)
"""
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from geo_distill import local_teacher
from geo_distill.cli import build_parser

NATIVE_DIM = 12


def vllm_args(extra=()):
    return build_parser().parse_args(
        ["local-teacher", "--backend", "vllm", "--no-push", *extra])


class FakeLLM:
    """Records constructor kwargs; embed() returns deterministic, deliberately
    UN-normalized vectors so the contract tests prove encode() normalizes."""

    last = None

    def __init__(self, **kwargs):
        if "runner" in kwargs:  # current API; task= is the old-pin fallback
            assert kwargs["runner"] == "pooling"
        FakeLLM.last = self
        self.kwargs = kwargs
        self.calls = []

    def embed(self, texts, use_tqdm=False, pooling_params=None):
        self.calls.append(list(texts))
        outs = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha1(t.encode()).digest()[:4], "little")
            vec = np.random.RandomState(seed).randn(NATIVE_DIM) * 3.0
            outs.append(SimpleNamespace(outputs=SimpleNamespace(
                embedding=vec.tolist())))
        return outs


@pytest.fixture()
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = FakeLLM
    # no PoolingParams attribute: exercises the older-API fallback branch
    monkeypatch.setitem(sys.modules, "vllm", mod)
    FakeLLM.last = None
    return mod


def test_dispatch_and_engine_kwargs(fake_vllm):
    args = vllm_args(["--max-seq-len", "384", "--tensor-parallel", "2",
                      "--gpu-memory-utilization", "0.9"])
    enc = local_teacher.make_encoder(args)
    assert getattr(enc, "owns_batching", False)
    kw = FakeLLM.last.kwargs
    assert kw["model"] == args.model and kw["tensor_parallel_size"] == 2
    assert kw["dtype"] == "float16" and kw["enforce_eager"] is True
    assert kw["max_model_len"] == 384
    assert kw["gpu_memory_utilization"] == pytest.approx(0.9)


def test_encode_output_contract(fake_vllm):
    enc = local_teacher.make_encoder(vllm_args(["--output-dim", "8"]))
    out = enc(["a", "b", "c"])
    assert out.shape == (3, 8) and out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    # row order follows input order, and truncation is Matryoshka-correct:
    # normalize the native vector, cut to 8 dims, renormalize
    raw = np.stack([FakeLLM(
    ).embed([t])[0].outputs.embedding for t in ["a", "b", "c"]]).astype(np.float32)
    full = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    want = full[:, :8] / np.linalg.norm(full[:, :8], axis=1, keepdims=True)
    np.testing.assert_allclose(out, want, atol=1e-6)


def test_output_dim_beyond_native_exits(fake_vllm):
    enc = local_teacher.make_encoder(vllm_args(["--output-dim", "999"]))
    with pytest.raises(SystemExit, match="exceeds"):
        enc(["a"])


def test_task_embed_fallback_for_old_pins(fake_vllm):
    class OldLLM(FakeLLM):
        def __init__(self, **kwargs):
            if "runner" in kwargs:
                raise TypeError("unexpected keyword argument 'runner'")
            super().__init__(**kwargs)

    fake_vllm.LLM = OldLLM
    local_teacher.make_encoder(vllm_args())
    assert FakeLLM.last.kwargs["task"] == "embed"


def test_missing_vllm_is_actionable(monkeypatch):
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    with pytest.raises(SystemExit, match="pip install vllm"):
        local_teacher.make_encoder(vllm_args())  # vLLM genuinely absent here


def test_spawn_env_is_set_before_import(fake_vllm, monkeypatch):
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("NCCL_IGNORE_DISABLED_P2P", raising=False)
    import os

    local_teacher.make_encoder(vllm_args())
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["NCCL_IGNORE_DISABLED_P2P"] == "1"


# ---------------------------------------------------------------------------
# cache-key separation + run() integration
# ---------------------------------------------------------------------------


def test_backend_separates_config_key_but_transformers_stays_stable():
    base = ("m", "passage", 8, 512, "", "chash")
    transformers_key = local_teacher.config_key(*base)
    assert local_teacher.config_key(*base, backend="transformers") == transformers_key
    assert local_teacher.config_key(*base, backend="vllm") != transformers_key
    # the transformers tag is byte-identical to the pre-backend format, so
    # every existing cache / pushed artifact still seeds and resumes
    legacy = hashlib.sha1("m|passage|8|512|\x00chash".encode()).hexdigest()[:16]
    assert transformers_key == legacy


def test_run_hands_whole_shards_to_owning_encoder(tmp_path, monkeypatch, sentences):
    (tmp_path / "corpus.txt").write_text("\n".join(sentences), encoding="utf-8")
    args = build_parser().parse_args([
        "local-teacher", "--backend", "vllm", "--no-push",
        "--input", str(tmp_path / "corpus.txt"),
        "--cache-dir", str(tmp_path / "cache"),
        "--out-emb", str(tmp_path / "teacher_emb.npy"),
        "--out-sents", str(tmp_path / "sentences.json"),
        "--out-meta", str(tmp_path / "teacher_meta.json"),
        "--checkpoint-every", "32", "--batch-size", "4",  # bs must be ignored
    ])

    calls = []

    def fake_encoder(texts):
        calls.append(list(texts))
        return np.eye(len(texts), NATIVE_DIM, dtype=np.float32)

    fake_encoder.owns_batching = True
    monkeypatch.setattr(local_teacher, "make_encoder", lambda a: fake_encoder)
    local_teacher.run(args)

    # one call per 32-sentence shard, regardless of --batch-size 4
    assert [len(c) for c in calls] == [32, 32, 32, 4]
    meta = json.loads((tmp_path / "teacher_meta.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "vllm" and meta["complete"] is True