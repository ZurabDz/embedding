"""Push/pull the teacher artifacts through a Hugging Face *dataset* repo.

The local-teacher stage runs where the GPUs are (Kaggle) while training may
happen in another session, so the three teacher artifacts round-trip through a
private dataset repo the same way lm round-trips checkpoints: `local-teacher
--push-to <repo>` uploads them after generation (partial runs included),
`fetch-teacher <repo>` restores them into artifacts/ so train/eval — and
local-teacher's own cross-session resume — pick them up unchanged.

The repo is machine-managed and holds exactly the newest snapshot; history is
squashed after every push because replaced LFS blobs (teacher_emb.npy is the
big one) would otherwise keep counting against the account quota.
"""
from __future__ import annotations

import json
import os
import shutil

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
