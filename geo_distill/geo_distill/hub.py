"""Push/pull the pipeline's two Hub round-trips: the teacher artifacts through
a *dataset* repo, and the student through a *model* repo.

Both exist for the same reason. The local-teacher stage runs where the GPUs are
(Kaggle) while training may happen in another session, and a training session is
itself capped long before ten epochs over a million sentences finish. So each
side round-trips through a private repo the way lm round-trips its checkpoints:

    local-teacher --push-to <repo>  ->  fetch-teacher <repo>      (dataset repo)
    train --push-to <repo>          ->  train --resume            (model repo)
                                    ->  fetch-student <repo>

Model and dataset ids are separate Hub namespaces, so every call here passes an
explicit repo_type; getting it wrong creates a stray repo of the other kind.

Both repos are machine-managed and hold exactly the newest snapshot; history is
squashed after every push because replaced LFS blobs (teacher_emb.npy and the
student's optimizer moments are the big ones) would otherwise keep counting
against the account quota.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from geo_distill import config as paths


def hf_token():
    # Lazy indirection: importing mlm pulls in jax, which this module (like
    # teacher.py) keeps out of its import chain.
    from mlm.hub import hf_token as _hf_token

    return _hf_token()


def strip_prefix(spec: str) -> str:
    from mlm.hub import strip_prefix as _strip_prefix

    return _strip_prefix(spec)

# Canonical filenames at the repo root — locals are renamed to these on push,
# whatever --out-* said, so fetch-teacher never has to guess.
FILES = ("teacher_emb.npy", "sentences.json", "teacher_meta.json")

# The student repo's contents. Unlike the teacher's, these already have fixed
# names inside a --out-dir, so the repo mirrors the directory one-for-one.
# TEACHER_MEAN only exists for a centered run and the two resume files only
# after the first checkpoint, so pushes and pulls both skip what is absent.
STUDENT_FILES = (paths.STUDENT_CONFIG, paths.STUDENT_PARAMS, paths.TEACHER_MEAN,
                 paths.STUDENT_STATE, paths.TRAIN_STATE)
# What a usable student needs; the rest is provenance and resume machinery.
STUDENT_REQUIRED = (paths.STUDENT_CONFIG, paths.STUDENT_PARAMS)
STUDENT_RESUME = (paths.STUDENT_STATE, paths.TRAIN_STATE)


def push_teacher_data(repo: str, emb_path: str, sents_path: str,
                      meta_path: str) -> None:
    """Upload the three teacher artifacts to a dataset repo, in one commit.

    The repo is created *private* on first push. A minimal dataset card is
    added only when the repo has none, so a hand-edited README is never
    clobbered.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    token = hf_token()
    if token is None:
        raise RuntimeError(
            "pushing teacher data needs a Hugging Face *write* token: set "
            "HF_TOKEN, or run `hf auth login` (tokens: hf.co/settings/tokens)"
        )
    repo = strip_prefix(repo)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=True)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    local = (emb_path, sents_path, meta_path)
    ops = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path)
           for name, path in zip(FILES, local)]
    if not api.file_exists(repo_id=repo, filename="README.md", repo_type="dataset"):
        card = "\n".join([
            "---", "pretty_name: Teacher embeddings (geo_distill)", "---", "",
            "Teacher embeddings for geo_distill, pushed by "
            "`geo-distill local-teacher` (machine-managed; newest snapshot only).", "",
            "- `teacher_emb.npy` — (N, D) L2-normalized vectors; row i pairs "
            "with sentence i",
            "- `sentences.json` — the N embedded sentences (flat JSON list)",
            "- `teacher_meta.json` — teacher model, dim, coverage, settings", "",
            "Consume in a training session:", "```bash",
            f"python -m geo_distill fetch-teacher {repo}", "```",
        ]) + "\n"
        ops.append(CommitOperationAdd(path_in_repo="README.md",
                                      path_or_fileobj=card.encode()))
    try:
        api.create_commit(
            repo_id=repo, repo_type="dataset", operations=ops,
            commit_message=(f"teacher embeddings: {meta.get('embedded')}/"
                            f"{meta.get('input_total')} ({meta.get('model')})"))
    except Exception as e:
        # create_repo(exist_ok=True) swallows the 403 when the repo id names
        # someone ELSE'S public repo, so no-write-access can first surface here.
        raise RuntimeError(
            f"upload to hf.co/datasets/{repo} failed: {e}\n"
            f"The token must have *write* scope and the namespace must be yours."
        ) from e
    try:
        # Replaced LFS blobs keep counting against the account quota (100 GB
        # free tier) until history is squashed — ~one teacher_emb.npy per push,
        # forever. This repo's history has no value: newest snapshot only.
        api.super_squash_history(repo_id=repo, repo_type="dataset")
    except Exception as e:  # noqa: BLE001
        print(f"  note: history squash failed ({e}); Hub storage grows by "
              f"~one teacher_emb.npy per push until it succeeds")
    total = sum(os.path.getsize(p) for p in local)
    print(f"pushed teacher data ({total / 1e6:.0f} MB) -> "
          f"hf.co/datasets/{repo} (private)")


def pull_teacher_data(repo: str, out_dir: str = paths.ARTIFACTS_DIR) -> dict:
    """Download the three artifacts into out_dir; returns the meta dict.

    Crash-safe like mlm.hub.pull_checkpoint: the download lands in a staging
    directory and files move into place (os.replace) only after the emb/
    sentences alignment is validated, so a killed or bad pull can never leave
    torn artifacts where train would find them.
    """
    from huggingface_hub import snapshot_download

    repo = strip_prefix(repo)
    os.makedirs(out_dir, exist_ok=True)
    staging = os.path.join(out_dir, ".pull-teacher")
    try:
        snapshot_download(repo_id=repo, repo_type="dataset", token=hf_token(),
                          local_dir=staging, allow_patterns=list(FILES))
    except Exception as e:
        raise RuntimeError(
            f"could not fetch teacher data from hf.co/datasets/{repo}: {e}\n"
            f"The repo should hold {', '.join(FILES)} (pushed by "
            f"`geo-distill local-teacher --push-to {repo}`). For a private "
            f"repo, set HF_TOKEN."
        ) from e
    for name in FILES:
        if not os.path.isfile(os.path.join(staging, name)):
            raise RuntimeError(
                f"hf.co/datasets/{repo} has no {name} — was it pushed by "
                f"`geo-distill local-teacher --push-to`?")
    # mmap keeps the (potentially GB-sized) matrix off the heap; only the
    # shape is needed here.
    emb = np.load(os.path.join(staging, FILES[0]), mmap_mode="r")
    with open(os.path.join(staging, FILES[1]), encoding="utf-8") as f:
        n_sents = len(json.load(f))
    if emb.ndim != 2 or emb.shape[0] != n_sents:
        raise RuntimeError(
            f"fetched teacher data is misaligned: embeddings {emb.shape} vs "
            f"{n_sents} sentences — the repo holds a torn push; re-push it")
    del emb
    for name in FILES:
        os.replace(os.path.join(staging, name), os.path.join(out_dir, name))
    shutil.rmtree(staging, ignore_errors=True)  # hf's .cache metadata remnants
    with open(os.path.join(out_dir, FILES[2]), encoding="utf-8") as f:
        return json.load(f)


def run_fetch(args) -> None:
    meta = pull_teacher_data(args.repo, args.out_dir)
    print(f"fetched {meta.get('embedded')}/{meta.get('input_total')} embeddings "
          f"(dim {meta.get('embedding_dim')}, model {meta.get('model')}, "
          f"coverage {(meta.get('coverage') or 0) * 100:.1f}%) -> {args.out_dir}")


# ----------------------------------------------------------------------------
# The student: a private *model* repo mirroring one `train --out-dir`
# ----------------------------------------------------------------------------


def ensure_student_repo(repo: str) -> None:
    """Create (or confirm) the student model repo before training starts.

    Worth knowing what this does and does not buy: it catches a typo'd or
    foreign namespace up front, but a read-scoped token on a repo that already
    exists and is readable passes here and only fails at the first push — the
    Hub's create_repo swallows the 403 in that case. It is a cheap filter, not
    a guarantee.
    """
    from mlm.hub import ensure_writable

    ensure_writable(repo, "model")


def hub_train_state(repo: str) -> dict | None:
    """The repo's train_state.json, or None when it holds no run yet.

    One small file rather than a snapshot: this is the "is the Hub ahead of
    local?" probe, and it runs before every resume and before every fresh push.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (EntryNotFoundError,
                                        RepositoryNotFoundError)

    try:
        path = hf_hub_download(repo_id=strip_prefix(repo), repo_type="model",
                               filename=paths.TRAIN_STATE, token=hf_token())
    except (EntryNotFoundError, RepositoryNotFoundError):
        # No such file, no such repo, or a private repo we cannot see: all three
        # mean "nothing to continue from here" as far as the caller cares.
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _student_card(repo: str) -> bytes:
    return "\n".join([
        "---", "library_name: flax", "---", "",
        "Georgian sentence-embedding student distilled from a teacher embedding "
        "model by [`geo_distill`](https://github.com/ZurabDz/embedding), pushed "
        "by `geo-distill train --push-to`.", "",
        f"- `{paths.STUDENT_CONFIG}` — architecture + provenance (which MLM "
        f"encoder and tokenizer it came from)",
        f"- `{paths.STUDENT_PARAMS}` — the parameters of the best epoch, by "
        f"held-out Spearman",
        f"- `{paths.TEACHER_MEAN}` — the training-split teacher mean; subtract "
        f"it from a raw teacher vector to map it into this student's space",
        f"- `{paths.STUDENT_STATE}` / `{paths.TRAIN_STATE}` — the newest "
        f"checkpoint's latest parameters, Adam moments and position, so "
        f"training continues across capped sessions", "",
        "Machine-managed: every push replaces the previous snapshot.", "",
        "Use it:", "```bash",
        f"python -m geo_distill fetch-student {repo}",
        "python -m geo_distill eval --model-dir artifacts", "```", "",
        "Continue training it:", "```bash",
        f"python -m geo_distill train ... --push-to {repo} --resume", "```",
    ]).encode() + b"\n"  # bytes, not str: a str path_or_fileobj is read as a *path*


def push_student(out_dir: str, repo: str, *, epoch: int, epochs: int,
                 best: float | None = None) -> None:
    """Upload one `train --out-dir` snapshot to the model repo, in one commit.

    Every file is replaced wholesale, so the repo always holds exactly one
    student and `--resume` never has to choose between versions. Commit
    operations are built fresh on each call: the Hub client marks them consumed
    once committed and refuses to reuse them.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    token = hf_token()
    if token is None:
        raise RuntimeError(
            "pushing the student needs a Hugging Face *write* token: set "
            "HF_TOKEN, or run `hf auth login` (tokens: hf.co/settings/tokens)"
        )
    repo = strip_prefix(repo)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True, private=True)

    ops, total = [], 0
    for name in STUDENT_FILES:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            ops.append(CommitOperationAdd(path_in_repo=name, path_or_fileobj=path))
            total += os.path.getsize(path)
    if not ops:
        raise RuntimeError(
            f"nothing to push: {os.path.abspath(out_dir)} holds none of "
            f"{', '.join(STUDENT_FILES)}")
    if not api.file_exists(repo_id=repo, repo_type="model", filename="README.md"):
        ops.append(CommitOperationAdd(path_in_repo="README.md",
                                      path_or_fileobj=_student_card(repo)))
    score = "" if best is None else f", val spearman {best:.3f}"
    t0 = time.time()
    try:
        api.create_commit(repo_id=repo, repo_type="model", operations=ops,
                          commit_message=f"student epoch {epoch}/{epochs}{score}")
    except Exception as e:
        # create_repo(exist_ok=True) swallows the 403 when the repo id names
        # someone ELSE'S readable repo, so no-write-access first surfaces here.
        raise RuntimeError(
            f"upload to hf.co/{repo} failed: {e}\n"
            f"The token must have *write* scope and the namespace must be yours."
        ) from e
    try:
        # Same reasoning as push_teacher_data: the replaced LFS blobs (the
        # optimizer moments are ~2x the parameters) keep counting against the
        # account quota until history is squashed, and this repo's history has
        # no value — newest snapshot only.
        api.super_squash_history(repo_id=repo, repo_type="model")
    except Exception as e:  # noqa: BLE001
        print(f"  note: history squash failed ({e}); Hub storage grows by "
              f"~one student snapshot per push until it succeeds")
    print(f"pushed student epoch {epoch}/{epochs} ({total / 1e6:.0f} MB) -> "
          f"hf.co/{repo} in {time.time() - t0:.0f}s")


def pull_student(repo: str, out_dir: str = paths.ARTIFACTS_DIR, *,
                 require_state: bool = False) -> dict:
    """Download the student artifacts into out_dir; returns the student config.

    Crash-safe like pull_teacher_data: the download lands in a staging
    directory (which also absorbs the .cache/huggingface bookkeeping the Hub
    client writes beside it) and files move into place only once everything
    expected is present, so a killed pull can never leave a config from one run
    next to the parameters of another.

    `require_state` additionally demands the resume pair — a `fetch-student` is
    happy with a finished student, a `--resume` is not.
    """
    from huggingface_hub import snapshot_download

    repo = strip_prefix(repo)
    os.makedirs(out_dir, exist_ok=True)
    staging = os.path.join(out_dir, ".pull-student")
    try:
        snapshot_download(repo_id=repo, repo_type="model", token=hf_token(),
                          local_dir=staging, allow_patterns=list(STUDENT_FILES))
    except Exception as e:
        raise RuntimeError(
            f"could not fetch the student from hf.co/{repo}: {e}\n"
            f"The repo should hold {', '.join(STUDENT_REQUIRED)} (pushed by "
            f"`geo-distill train --push-to {repo}`). For a private repo, set "
            f"HF_TOKEN or run `hf auth login`."
        ) from e
    for name in STUDENT_REQUIRED:
        if not os.path.isfile(os.path.join(staging, name)):
            raise RuntimeError(
                f"hf.co/{repo} has no {name} — was it pushed by "
                f"`geo-distill train --push-to {repo}`?")
    if require_state:
        for name in STUDENT_RESUME:
            if not os.path.isfile(os.path.join(staging, name)):
                raise RuntimeError(
                    f"hf.co/{repo} has no {name}: it holds a student but no "
                    f"resume state, so there is nothing to continue from. "
                    f"Start a fresh run (drop --resume), which will replace it.")
    with open(os.path.join(staging, paths.STUDENT_CONFIG), encoding="utf-8") as f:
        cfg = json.load(f)
    for name in STUDENT_FILES:
        src = os.path.join(staging, name)
        if os.path.isfile(src):
            os.replace(src, os.path.join(out_dir, name))
    shutil.rmtree(staging, ignore_errors=True)  # hf's .cache metadata remnants
    return cfg


def pull_student_for_resume(out_dir: str, repo: str) -> None:
    """Restore the repo's checkpoint into out_dir when it is ahead of local.

    Local wins ties, so an uninterrupted local run never re-downloads; when
    neither side has a checkpoint this raises rather than letting --resume die
    later on a missing-file error. The mirror of mlm.hub.pull_checkpoint.
    """
    remote = hub_train_state(repo)
    local_state = None
    local_path = os.path.join(out_dir, paths.TRAIN_STATE)
    if os.path.isfile(local_path):
        with open(local_path, encoding="utf-8") as f:
            local_state = json.load(f)
    local = None if local_state is None else local_state.get("epoch")
    hub_epoch = None if remote is None else remote.get("epoch")
    if hub_epoch is None and local is None:
        raise RuntimeError(
            f"no student checkpoint on hf.co/{strip_prefix(repo)} or in "
            f"{os.path.abspath(out_dir)} yet — this looks like the first "
            f"session, so run the same command without --resume")
    if hub_epoch is None or (local is not None and local >= hub_epoch):
        return
    # The local twin of train.py's fresh-run guard, and it has to happen here:
    # promoting the repo's files unlinks this directory's, so two different
    # runs have to be told apart while both are still on disk. train.py's
    # fingerprint check would otherwise refuse the resume *after* the local
    # run's best parameters had already been overwritten by the remote ones.
    # --epochs is deliberately absent from the fingerprint, so a legitimately
    # chained session matches here; only a --push-to naming another run does not.
    remote_fp, local_fp = remote.get("fingerprint"), (local_state or {}).get("fingerprint")
    if remote_fp and local_fp and remote_fp != local_fp:
        raise RuntimeError(
            f"hf.co/{strip_prefix(repo)} holds a different run (epoch "
            f"{hub_epoch}) than {os.path.abspath(out_dir)} (epoch {local}), so "
            f"pulling it would replace this run's parameters. Point --push-to "
            f"at the repo this run belongs to, or drop --push-to to continue "
            f"locally.")
    print(f"pulling student checkpoint (epoch {hub_epoch}) from "
          f"hf.co/{strip_prefix(repo)} ...", flush=True)
    pull_student(repo, out_dir, require_state=True)


def run_fetch_student(args) -> None:
    cfg = pull_student(args.repo, args.out_dir)
    best = cfg.get("best_val_spearman")
    state = None
    path = os.path.join(args.out_dir, paths.TRAIN_STATE)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    where = f"{cfg.get('student_type')} student, out_dim {cfg.get('out_dim')}"
    score = "" if best is None else f", best val spearman {best:.3f}"
    epochs = "" if state is None else (f", epoch {state.get('epoch')}/"
                                       f"{state.get('epochs')} (resumable)")
    print(f"fetched {where}{score}{epochs} -> {args.out_dir}")
    print(f"  score it with: python -m geo_distill eval --model-dir {args.out_dir}")
