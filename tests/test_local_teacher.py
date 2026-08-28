"""local-teacher offline contract: shard cache round-trips, resume, seeding,
OOM degradation, and the Hub dataset push/pull — all with a fake encoder and a
fake HfApi, so the suite never needs torch, a model download, or the network.

Args are built through the real CLI parser so these tests cannot drift from
the flags `run()` actually receives.
"""
import hashlib
import json
import shutil

import numpy as np
import pytest

from geo_distill import local_teacher
from geo_distill.cli import build_parser

DIM = 16


def _vec(text: str) -> np.ndarray:
    """Deterministic unit vector per (formatted) text."""
    seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "little")
    v = np.random.RandomState(seed).randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEncoder:
    """Stands in for make_encoder(args): deterministic vectors, full call log,
    and optional failure injection (KeyboardInterrupt / OOM)."""

    def __init__(self, interrupt_after=None, oom_above=None):
        self.batches = []
        self.produced = 0
        self.interrupt_after = interrupt_after
        self.oom_above = oom_above

    def __call__(self, texts):
        if self.oom_above is not None and len(texts) > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        if (self.interrupt_after is not None
                and self.produced >= self.interrupt_after):
            raise KeyboardInterrupt
        self.batches.append(list(texts))
        self.produced += len(texts)
        return np.stack([_vec(t) for t in texts])


def make_args(tmp_path, extra=()):
    return build_parser().parse_args([
        "local-teacher", "--no-push",
        "--input", str(tmp_path / "corpus.txt"),
        "--cache-dir", str(tmp_path / "cache"),
        "--out-emb", str(tmp_path / "teacher_emb.npy"),
        "--out-sents", str(tmp_path / "sentences.json"),
        "--out-meta", str(tmp_path / "teacher_meta.json"),
        "--checkpoint-every", "32",
        *extra,
    ])


@pytest.fixture()
def corpus(tmp_path, sentences):
    (tmp_path / "corpus.txt").write_text("\n".join(sentences), encoding="utf-8")
    return sentences


def run_with(monkeypatch, args, encoder):
    monkeypatch.setattr(local_teacher, "make_encoder", lambda a: encoder)
    local_teacher.run(args)
    return encoder


def test_full_run_writes_aligned_artifacts(tmp_path, monkeypatch, corpus):
    args = make_args(tmp_path)
    run_with(monkeypatch, args, FakeEncoder())

    emb = np.load(args.out_emb)
    assert emb.shape == (len(corpus), DIM) and emb.dtype == np.float32
    # rows went through the fp16 shard cache, so unit norm only up to fp16 eps
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=2e-3)
    assert json.loads((tmp_path / "sentences.json").read_text(encoding="utf-8")) == corpus

    meta = json.loads((tmp_path / "teacher_meta.json").read_text(encoding="utf-8"))
    gemini_meta_keys = {"model", "input_type", "output_dim_requested",
                        "embedding_dim", "input_file", "input_total",
                        "embedded", "coverage", "complete", "created"}
    assert gemini_meta_keys <= set(meta)
    assert meta["complete"] is True and meta["embedded"] == len(corpus)
    assert meta["config_key"]  # the extra that enables cross-session seeding

    shard_dir = tmp_path / "cache" / meta["config_key"]
    starts = sorted(int(p.stem.split("_")[1]) for p in shard_dir.glob("shard_*.npy"))
    assert starts == [0, 32, 64, 96]
    assert np.load(shard_dir / "shard_00000000.npy").dtype == np.float16


def test_resume_skips_completed_shards(tmp_path, monkeypatch, corpus):
    args = make_args(tmp_path)
    run_with(monkeypatch, args, FakeEncoder())
    first = np.load(args.out_emb)

    second = run_with(monkeypatch, make_args(tmp_path), FakeEncoder())
    assert second.batches == []  # fully cached: the model is never even loaded
    np.testing.assert_array_equal(np.load(args.out_emb), first)


def test_config_key_invalidation(tmp_path, monkeypatch, corpus):
    run_with(monkeypatch, make_args(tmp_path), FakeEncoder())
    # different --output-dim -> different key -> full re-encode in a new subdir
    enc = run_with(monkeypatch, make_args(tmp_path, ["--output-dim", "8"]),
                   FakeEncoder())
    assert sum(len(b) for b in enc.batches) == len(corpus)
    assert len(list((tmp_path / "cache").iterdir())) == 2

    # a changed corpus also changes the key (via corpus_sha1)
    (tmp_path / "corpus.txt").write_text("\n".join(corpus[:-1]), encoding="utf-8")
    enc = run_with(monkeypatch, make_args(tmp_path), FakeEncoder())
    assert sum(len(b) for b in enc.batches) == len(corpus) - 1
    assert len(list((tmp_path / "cache").iterdir())) == 3


def test_interrupt_finalizes_partial_and_resumes(tmp_path, monkeypatch, corpus):
    args = make_args(tmp_path)
    run_with(monkeypatch, args, FakeEncoder(interrupt_after=64))

    emb = np.load(args.out_emb)
    assert emb.shape == (64, DIM)
    assert json.loads((tmp_path / "sentences.json").read_text(encoding="utf-8")) == corpus[:64]
    meta = json.loads((tmp_path / "teacher_meta.json").read_text(encoding="utf-8"))
    assert meta["complete"] is False and meta["coverage"] == pytest.approx(0.64)

    enc = run_with(monkeypatch, make_args(tmp_path), FakeEncoder())
    assert sum(len(b) for b in enc.batches) == 36  # only the missing tail
    assert np.load(args.out_emb).shape == (len(corpus), DIM)


def test_seed_from_fetched_outputs(tmp_path, monkeypatch, corpus):
    artifact_names = ("teacher_emb.npy", "sentences.json", "teacher_meta.json")
    args = make_args(tmp_path)
    run_with(monkeypatch, args, FakeEncoder(interrupt_after=64))
    partial = {n: (tmp_path / n).read_bytes() for n in artifact_names}

    run_with(monkeypatch, make_args(tmp_path), FakeEncoder())
    full = np.load(args.out_emb)

    # new session: fetch-teacher restored the PARTIAL artifacts, but the shard
    # cache is gone -> seeding rebuilds the covered shards instead of re-encoding
    for name, data in partial.items():
        (tmp_path / name).write_bytes(data)
    shutil.rmtree(tmp_path / "cache")
    enc = run_with(monkeypatch, make_args(tmp_path), FakeEncoder())
    assert sum(len(b) for b in enc.batches) == 36  # 64 rows came from seeding
    np.testing.assert_array_equal(np.load(args.out_emb), full)

    # a Gemini-shaped meta (no config_key) must never seed anything
    meta = json.loads(partial["teacher_meta.json"])
    del meta["config_key"]
    (tmp_path / "teacher_emb.npy").write_bytes(partial["teacher_emb.npy"])
    (tmp_path / "sentences.json").write_bytes(partial["sentences.json"])
    (tmp_path / "teacher_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.rmtree(tmp_path / "cache")
    enc = run_with(monkeypatch, make_args(tmp_path), FakeEncoder())
    assert sum(len(b) for b in enc.batches) == len(corpus)


def test_query_instruction_formatting(tmp_path, corpus):
    assert local_teacher.format_text("abc", "passage", "") == "abc"
    assert (local_teacher.format_text("abc", "query", "find it")
            == "Instruct: find it\nQuery:abc")

    # the instruction is part of the config key for queries only
    def key(input_type, instruction_flag):
        a = make_args(tmp_path, ["--input-type", input_type]
                      + (["--instruction", instruction_flag] if instruction_flag else []))
        instr = local_teacher.resolve_instruction(a)
        return local_teacher.config_key(a.model, a.input_type, a.output_dim,
                                        a.max_seq_len, instr, "chash")

    assert key("query", None) != key("query", "custom instruction")
    assert key("passage", None) == key("passage", "custom instruction")
    assert key("passage", None) != key("query", None)


def test_oom_halves_batch_persistently(tmp_path, monkeypatch, corpus):
    args = make_args(tmp_path)  # default --batch-size 16
    enc = run_with(monkeypatch, args, FakeEncoder(oom_above=4))
    assert np.load(args.out_emb).shape == (len(corpus), DIM)
    assert max(len(b) for b in enc.batches) <= 4  # halved 16->8->4, never grew


def test_cli_wiring_defaults():
    a = build_parser().parse_args(["local-teacher"])
    assert a.model == "Qwen/Qwen3-Embedding-8B"
    assert a.output_dim == 1024 and a.store_dtype == "float32"
    assert a.device_map == "auto" and a.checkpoint_every == 2048
    assert a.push_to == "ZurabDz/geo-teacher-qwen3-8b" and not a.no_push
    f = build_parser().parse_args(["fetch-teacher"])
    assert f.repo == "ZurabDz/geo-teacher-qwen3-8b"
    f = build_parser().parse_args(["fetch-teacher", "user/other"])
    assert f.repo == "user/other"


# ---------------------------------------------------------------------------
# Hub dataset push/pull (fake HfApi / snapshot_download — no network)
# ---------------------------------------------------------------------------


class FakeApi:
    def __init__(self, *, readme_exists=False):
        self.calls = []
        self.readme_exists = readme_exists

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def file_exists(self, **kw):
        self.calls.append(("file_exists", kw))
        return self.readme_exists

    def create_commit(self, **kw):
        self.calls.append(("create_commit", kw))

    def super_squash_history(self, **kw):
        self.calls.append(("super_squash_history", kw))


def _write_artifacts(d, n=5, dim=4):
    emb = np.random.RandomState(0).randn(n, dim).astype(np.float32)
    np.save(d / "teacher_emb.npy", emb)
    (d / "sentences.json").write_text(json.dumps([f"s{i}" for i in range(n)]),
                                      encoding="utf-8")
    (d / "teacher_meta.json").write_text(json.dumps(
        {"embedded": n, "input_total": n, "model": "m", "embedding_dim": dim,
         "coverage": 1.0}), encoding="utf-8")
    return emb


def test_push_teacher_data_dataset_repo(tmp_path, monkeypatch):
    import huggingface_hub

    from geo_distill import hub as gh

    _write_artifacts(tmp_path)
    api = FakeApi()
    monkeypatch.setattr(gh, "hf_token", lambda: "tok")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)

    gh.push_teacher_data("u/r", str(tmp_path / "teacher_emb.npy"),
                         str(tmp_path / "sentences.json"),
                         str(tmp_path / "teacher_meta.json"))

    by_name = dict(api.calls)
    assert by_name["create_repo"]["repo_type"] == "dataset"
    assert by_name["create_repo"]["private"] is True
    commit = by_name["create_commit"]
    assert commit["repo_type"] == "dataset"
    names = [op.path_in_repo for op in commit["operations"]]
    assert names == list(gh.FILES) + ["README.md"]  # first push adds the card
    assert by_name["super_squash_history"]["repo_type"] == "dataset"

    # an existing README is never clobbered
    api2 = FakeApi(readme_exists=True)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api2)
    gh.push_teacher_data("u/r", str(tmp_path / "teacher_emb.npy"),
                         str(tmp_path / "sentences.json"),
                         str(tmp_path / "teacher_meta.json"))
    ops = dict(api2.calls)["create_commit"]["operations"]
    assert [op.path_in_repo for op in ops] == list(gh.FILES)


def test_push_without_token_fails_fast(tmp_path, monkeypatch):
    from geo_distill import hub as gh

    _write_artifacts(tmp_path)
    monkeypatch.setattr(gh, "hf_token", lambda: None)
    with pytest.raises(RuntimeError, match="write.*token"):
        gh.push_teacher_data("u/r", str(tmp_path / "teacher_emb.npy"),
                             str(tmp_path / "sentences.json"),
                             str(tmp_path / "teacher_meta.json"))


def test_pull_teacher_data_roundtrip_and_validation(tmp_path, monkeypatch):
    import huggingface_hub

    from geo_distill import hub as gh

    src = tmp_path / "src"
    src.mkdir()
    emb = _write_artifacts(src)

    def fake_snapshot(repo_id, repo_type, token, local_dir, allow_patterns):
        assert repo_type == "dataset" and set(allow_patterns) == set(gh.FILES)
        shutil.copytree(src, local_dir, dirs_exist_ok=True)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    out = tmp_path / "artifacts"
    meta = gh.pull_teacher_data("u/r", str(out))
    assert meta["embedded"] == 5
    np.testing.assert_array_equal(np.load(out / "teacher_emb.npy"), emb)
    assert not (out / ".pull-teacher").exists()  # staging cleaned up

    # a repo missing an artifact fails loudly
    (src / "teacher_emb.npy").unlink()
    with pytest.raises(RuntimeError, match="has no teacher_emb.npy"):
        gh.pull_teacher_data("u/r", str(tmp_path / "a2"))

    # misaligned emb/sentences never lands in out_dir
    np.save(src / "teacher_emb.npy", emb[:3])
    with pytest.raises(RuntimeError, match="misaligned"):
        gh.pull_teacher_data("u/r", str(tmp_path / "a3"))
    assert not (tmp_path / "a3" / "teacher_emb.npy").exists()
