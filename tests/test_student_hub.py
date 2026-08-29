"""Student round-trip through a HF *model* repo: `train --push-to`, the pull
`--resume` does, and `fetch-student`.

Model and dataset ids are separate Hub namespaces, so the repo_type on every
call is a contract, not a detail — the teacher artifacts next door use
"dataset". Nothing here touches the network: HfApi and snapshot_download are
faked the same way tests/test_local_teacher.py fakes them, which works because
hub.py imports them inside the functions and resolves hf_token through a
module-level indirection.
"""
import json
import shutil

import numpy as np
import pytest

from geo_distill import config as paths
from geo_distill.cli import build_parser


class FakeApi:
    def __init__(self, *, readme_exists=False, fail_commit=False):
        self.calls = []
        self.readme_exists = readme_exists
        self.fail_commit = fail_commit

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def file_exists(self, **kw):
        self.calls.append(("file_exists", kw))
        return self.readme_exists

    def create_commit(self, **kw):
        self.calls.append(("create_commit", kw))
        if self.fail_commit:
            raise OSError("boom")

    def super_squash_history(self, **kw):
        self.calls.append(("super_squash_history", kw))


def _write_student(d, *, epoch=2, epochs=4, resumable=True, centered=True):
    """A --out-dir as train leaves it. Written through json/numpy directly
    rather than by training, because what is under test is which *files* move."""
    d.mkdir(parents=True, exist_ok=True)
    (d / paths.STUDENT_CONFIG).write_text(json.dumps(
        {"student_type": "scratch", "vocab_size": 500, "out_dim": 8,
         "max_len": 16, "tokenizer": "t.json", "center": centered,
         "best_val_spearman": 0.5}), encoding="utf-8")
    (d / paths.STUDENT_PARAMS).write_bytes(b"\x80params")
    if centered:
        np.save(d / paths.TEACHER_MEAN, np.zeros((1, 8), np.float32))
    if resumable:
        (d / paths.STUDENT_STATE).write_bytes(b"\x80state")
        (d / paths.TRAIN_STATE).write_text(json.dumps(
            {"epoch": epoch, "epochs": epochs, "best_val_spearman": 0.5}),
            encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# Push
# --------------------------------------------------------------------------- #
def test_push_student_uses_a_model_repo(tmp_path, monkeypatch):
    import huggingface_hub

    from geo_distill import hub as gh

    _write_student(tmp_path / "student")
    api = FakeApi()
    monkeypatch.setattr(gh, "hf_token", lambda: "tok")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)

    gh.push_student(str(tmp_path / "student"), "u/r", epoch=2, epochs=4, best=0.5)

    by_name = dict(api.calls)
    assert by_name["create_repo"]["repo_type"] == "model"
    assert by_name["create_repo"]["private"] is True
    commit = by_name["create_commit"]
    assert commit["repo_type"] == "model"
    assert "epoch 2/4" in commit["commit_message"]
    names = [op.path_in_repo for op in commit["operations"]]
    assert names == list(gh.STUDENT_FILES) + ["README.md"]
    assert by_name["super_squash_history"]["repo_type"] == "model"

    # an existing card is never clobbered
    api2 = FakeApi(readme_exists=True)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api2)
    gh.push_student(str(tmp_path / "student"), "u/r", epoch=3, epochs=4)
    ops = dict(api2.calls)["create_commit"]["operations"]
    assert [op.path_in_repo for op in ops] == list(gh.STUDENT_FILES)


def test_push_student_skips_files_that_do_not_exist(tmp_path, monkeypatch):
    """A --no-center run has no teacher_mean, and nothing has resume state
    before the first checkpoint; neither may abort the push."""
    import huggingface_hub

    from geo_distill import hub as gh

    _write_student(tmp_path / "student", resumable=False, centered=False)
    api = FakeApi(readme_exists=True)
    monkeypatch.setattr(gh, "hf_token", lambda: "tok")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token: api)

    gh.push_student(str(tmp_path / "student"), "u/r", epoch=1, epochs=1)
    ops = dict(api.calls)["create_commit"]["operations"]
    assert [op.path_in_repo for op in ops] == list(gh.STUDENT_REQUIRED)


def test_push_student_without_token_fails_fast(tmp_path, monkeypatch):
    from geo_distill import hub as gh

    _write_student(tmp_path / "student")
    monkeypatch.setattr(gh, "hf_token", lambda: None)
    with pytest.raises(RuntimeError, match="write.*token"):
        gh.push_student(str(tmp_path / "student"), "u/r", epoch=1, epochs=1)


def test_push_student_reports_the_write_scope_on_failure(tmp_path, monkeypatch):
    """create_repo(exist_ok=True) swallows the 403 for a readable repo you
    cannot write, so no-write-access first surfaces at the commit."""
    import huggingface_hub

    from geo_distill import hub as gh

    _write_student(tmp_path / "student")
    monkeypatch.setattr(gh, "hf_token", lambda: "tok")
    monkeypatch.setattr(huggingface_hub, "HfApi",
                        lambda token: FakeApi(fail_commit=True))
    with pytest.raises(RuntimeError, match="write.*scope"):
        gh.push_student(str(tmp_path / "student"), "u/r", epoch=1, epochs=1)


# --------------------------------------------------------------------------- #
# Pull
# --------------------------------------------------------------------------- #
def _fake_snapshot(src, monkeypatch, *, seen=None):
    import huggingface_hub

    def fake(repo_id, repo_type, token, local_dir, allow_patterns):
        assert repo_type == "model"
        if seen is not None:
            seen.append(set(allow_patterns))
        shutil.copytree(src, local_dir, dirs_exist_ok=True)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)


def test_pull_student_roundtrip_and_staging_cleanup(tmp_path, monkeypatch):
    from geo_distill import hub as gh

    src = _write_student(tmp_path / "src")
    seen = []
    _fake_snapshot(src, monkeypatch, seen=seen)
    monkeypatch.setattr(gh, "hf_token", lambda: None)

    out = tmp_path / "artifacts"
    cfg = gh.pull_student("u/r", str(out))

    assert seen == [set(gh.STUDENT_FILES)]
    assert cfg["student_type"] == "scratch"
    for name in gh.STUDENT_FILES:
        assert (out / name).is_file()
    assert not (out / ".pull-student").exists()   # staging + hf's .cache removed


def test_pull_student_requires_the_essentials(tmp_path, monkeypatch):
    from geo_distill import hub as gh

    src = _write_student(tmp_path / "src")
    (src / paths.STUDENT_PARAMS).unlink()
    _fake_snapshot(src, monkeypatch)
    monkeypatch.setattr(gh, "hf_token", lambda: None)
    with pytest.raises(RuntimeError, match="has no student_params.msgpack"):
        gh.pull_student("u/r", str(tmp_path / "out"))


def test_pull_student_for_resume_wants_the_resume_pair(tmp_path, monkeypatch):
    """fetch-student is happy with a finished student; --resume is not."""
    from geo_distill import hub as gh

    src = _write_student(tmp_path / "src", resumable=False)
    _fake_snapshot(src, monkeypatch)
    monkeypatch.setattr(gh, "hf_token", lambda: None)

    gh.pull_student("u/r", str(tmp_path / "fetch"))                  # fine
    with pytest.raises(RuntimeError, match="nothing to continue from"):
        gh.pull_student("u/r", str(tmp_path / "resume"), require_state=True)


def test_resume_pull_prefers_local_when_it_is_not_behind(tmp_path, monkeypatch):
    """Local wins ties, so an uninterrupted local run never re-downloads."""
    from geo_distill import hub as gh

    out = _write_student(tmp_path / "out", epoch=5)
    pulled = []
    monkeypatch.setattr(gh, "pull_student",
                        lambda *a, **k: pulled.append(a) or {})

    monkeypatch.setattr(gh, "hub_train_state", lambda repo: {"epoch": 5})
    gh.pull_student_for_resume(str(out), "u/r")
    assert pulled == []                                  # tie -> local wins

    monkeypatch.setattr(gh, "hub_train_state", lambda repo: {"epoch": 3})
    gh.pull_student_for_resume(str(out), "u/r")
    assert pulled == []                                  # local ahead

    monkeypatch.setattr(gh, "hub_train_state", lambda repo: {"epoch": 9})
    gh.pull_student_for_resume(str(out), "u/r")
    assert len(pulled) == 1                              # Hub ahead -> pull


def test_resume_pull_refuses_to_replace_a_different_run(tmp_path, monkeypatch):
    """The pull promotes files over the local ones, so it has to tell two runs
    apart *before* anything moves — train.py's fingerprint check would
    otherwise refuse the resume after the local best parameters were gone."""
    from geo_distill import hub as gh

    out = _write_student(tmp_path / "out", epoch=2)
    (out / paths.TRAIN_STATE).write_text(json.dumps(
        {"epoch": 2, "epochs": 6, "fingerprint": {"seed": 0, "dim": 32}}),
        encoding="utf-8")
    mine = (out / paths.STUDENT_PARAMS).read_bytes()
    other = _write_student(tmp_path / "other")
    (other / paths.STUDENT_PARAMS).write_bytes(b"\x80theirs")  # distinguishable
    _fake_snapshot(other, monkeypatch)
    monkeypatch.setattr(gh, "hf_token", lambda: None)

    monkeypatch.setattr(gh, "hub_train_state", lambda repo: {
        "epoch": 5, "fingerprint": {"seed": 5, "dim": 64}})
    with pytest.raises(RuntimeError, match="holds a different run"):
        gh.pull_student_for_resume(str(out), "u/r")
    assert (out / paths.STUDENT_PARAMS).read_bytes() == mine   # untouched

    # the same run further along is exactly what --resume is for; --epochs is
    # not in the fingerprint, so a chained session's two sides match
    monkeypatch.setattr(gh, "hub_train_state", lambda repo: {
        "epoch": 5, "fingerprint": {"seed": 0, "dim": 32}})
    gh.pull_student_for_resume(str(out), "u/r")
    assert (out / paths.STUDENT_PARAMS).read_bytes() != mine   # pulled

    # a repo pushed before fingerprints existed still pulls (nothing to compare)
    out2 = _write_student(tmp_path / "out2", epoch=1)
    monkeypatch.setattr(gh, "hub_train_state", lambda repo: {"epoch": 5})
    gh.pull_student_for_resume(str(out2), "u/r")


def test_resume_pull_with_nothing_anywhere_is_actionable(tmp_path, monkeypatch):
    from geo_distill import hub as gh

    monkeypatch.setattr(gh, "hub_train_state", lambda repo: None)
    with pytest.raises(RuntimeError, match="without --resume"):
        gh.pull_student_for_resume(str(tmp_path / "empty"), "u/r")


def test_hub_train_state_is_none_when_the_repo_has_no_run(monkeypatch):
    """A missing file, a missing repo and a private repo we cannot see all mean
    'nothing to continue from' — and they raise three different exceptions."""
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import (EntryNotFoundError,
                                        RemoteEntryNotFoundError,
                                        RepositoryNotFoundError)

    from geo_distill import hub as gh

    resp = httpx.Response(404, request=httpx.Request("GET", "https://hf.co"))
    monkeypatch.setattr(gh, "hf_token", lambda: None)
    for exc in (EntryNotFoundError("no file"),
                RemoteEntryNotFoundError("no file", response=resp),
                RepositoryNotFoundError("no repo", response=resp)):
        def raiser(*a, _e=exc, **k):
            raise _e

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", raiser)
        assert gh.hub_train_state("u/r") is None


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def test_cli_wiring_defaults():
    a = build_parser().parse_args(["train"])
    assert a.push_to == "" and a.push_every == 0        # pushing is opt-in
    assert a.save_every == 1 and a.resume is False

    f = build_parser().parse_args(["fetch-student"])
    assert f.repo == paths.DEFAULT_STUDENT_MODEL_REPO
    assert f.out_dir == paths.ARTIFACTS_DIR
    f = build_parser().parse_args(["fetch-student", "user/other"])
    assert f.repo == "user/other"


@pytest.mark.parametrize("argv,needle", [
    (["train", "--push-every", "2"], "needs --push-to"),
    (["train", "--push-to", "u/r", "--push-every", "2", "--save-every", "0"],
     "needs --save-every"),
    (["train", "--push-to", "u/r", "--push-every", "3", "--save-every", "2"],
     "divisor"),
])
def test_push_flag_combinations_are_rejected_up_front(monkeypatch, argv, needle,
                                                      capsys):
    """Before the corpus load and the encoder download, not an epoch in."""
    from geo_distill import cli, hub

    monkeypatch.setattr(hub, "hf_token", lambda: "tok")
    monkeypatch.setattr(hub, "ensure_student_repo", lambda repo: None)
    with pytest.raises(SystemExit):
        cli.main(argv)
    assert needle in capsys.readouterr().err


def test_push_to_without_a_token_is_rejected_up_front(monkeypatch, capsys):
    from geo_distill import cli, hub

    monkeypatch.setattr(hub, "hf_token", lambda: None)
    with pytest.raises(SystemExit):
        cli.main(["train", "--push-to", "u/r"])
    assert "write token" in capsys.readouterr().err


def test_a_fresh_run_refuses_to_replace_a_repo_holding_a_run(tmp_path, monkeypatch):
    """Unlike the local directory, the repo's checkpoint has no copy anywhere
    else, so a fresh push over it is refused rather than warned about."""
    from geo_distill import hub
    from geo_distill.train import _resume_prelude

    monkeypatch.setattr(hub, "hub_train_state",
                        lambda repo: {"epoch": 3, "epochs": 10,
                                      "best_val_spearman": 0.6})
    args = build_parser().parse_args(["train", "--push-to", "u/r"])
    with pytest.raises(RuntimeError, match="Pass --resume to continue it"):
        _resume_prelude(args)

    monkeypatch.setattr(hub, "hub_train_state", lambda repo: None)
    assert _resume_prelude(args) is None            # empty repo: go ahead
