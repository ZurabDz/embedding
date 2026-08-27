"""Hugging Face Hub integration for the tokenizer.

Tokenizers: push a trained BPE as tokenizer.json / load one by repo id
(--tokenizer-path). Checkpoints: --hub-checkpoints syncs orbax checkpoints —
each push atomically replaces the repo's previous checkpoint (the repo holds
exactly one, the newest, under <step>/ plus config.json, mirroring the local
--save-dir layout), and --resume pulls it back when it is newer than anything
local, so the same training command chains across capped Kaggle/Colab
sessions.

Authentication follows huggingface_hub's own resolution order: the Colab
secrets vault, then the HF_TOKEN env var, then the token file written by
`hf auth login` / login(). Pushing without a usable *write* token fails up
front, before any corpus work; loading a public repo needs no token at all.
"""

import os
import re
import shutil
import time

# user-or-org / repo-name, the only shape the Hub accepts
_HUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+")


def hf_token() -> str | None:
    """The Hub token, resolved exactly as huggingface_hub itself would.

    get_token() checks the Colab secrets vault, then HF_TOKEN /
    HUGGING_FACE_HUB_TOKEN, then the file `hf auth login` writes — so a user
    who logged in in a notebook cell is not refused a push they could make.
    """
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def strip_prefix(spec: str) -> str:
    return spec.removeprefix("hf://")


def is_hub_id(spec: str) -> bool:
    """True when `spec` names a Hub repo rather than a filesystem path.

    hf://user/repo is explicit and always wins. A bare user/repo also counts,
    but only when it could not plausibly be a training target: exactly one
    slash, a valid repo-id shape, not dot-/slash-prefixed, and no .json
    suffix — local tokenizer files are JSON by convention (the default is
    ka_bpe.json), so `data/tok.json` stays a path and never hits the network.
    """
    if spec.startswith("hf://"):
        return True
    if spec.startswith((".", "/", "~")) or spec.endswith(".json"):
        return False
    return spec.count("/") == 1 and bool(_HUB_ID.fullmatch(spec))


def load_hub_tokenizer(spec: str):
    """tokenizer.json from a Hub repo -> a ready Tokenizer.

    Downloaded through huggingface_hub (cached under ~/.cache/huggingface, so
    a resumed session does not re-fetch) rather than tokenizers'
    from_pretrained, so private repos work through the same HF_TOKEN and the
    failure mode is one clear error instead of a Rust panic.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    repo = strip_prefix(spec)
    try:
        local = hf_hub_download(repo_id=repo, filename="tokenizer.json",
                                token=hf_token())
    except Exception as e:
        raise RuntimeError(
            f"--tokenizer-path {spec} looks like a Hub repo id, but "
            f"tokenizer.json could not be fetched from hf.co/{repo}: {e}\n"
            f"For a private repo, set HF_TOKEN. For a local file, use a path "
            f"that exists or ends in .json."
        ) from e
    tok = Tokenizer.from_file(local)
    print(f"loaded tokenizer from hf.co/{repo} (vocab {tok.get_vocab_size()})")
    return tok


def ensure_writable(repo: str) -> None:
    """Create (or confirm) the target repo in one cheap round-trip, up front.

    Token *presence* is not enough: Kaggle/Colab HF tokens are often read-scoped
    (made for gated datasets), and a 403 would otherwise surface only at push
    time, after the sample download and BPE training. create_repo validates both
    write scope and namespace ownership; it is idempotent, so push_tokenizer's
    later create_repo is a no-op. Side effect worth knowing: the (private) repo
    exists even if the run later dies before pushing anything.
    """
    from huggingface_hub import HfApi

    repo = strip_prefix(repo)
    try:
        HfApi(token=hf_token()).create_repo(repo_id=repo, exist_ok=True, private=True)
    except Exception as e:
        raise RuntimeError(
            f"cannot push to hf.co/{repo}: {e}\n"
            f"The token must have *write* scope (hf.co/settings/tokens) and "
            f"the namespace must be yours."
        ) from e


def push_tokenizer(path: str, repo: str, details: dict | None = None) -> None:
    """Upload `path` as tokenizer.json to a Hub repo, in one commit.

    The repo is created *private* on first push — flip it public in the repo
    settings when it is ready to share. A minimal model card is added only
    when the repo has none, so a hand-edited README is never clobbered.
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    token = hf_token()
    if token is None:
        raise RuntimeError(
            "pushing the tokenizer needs a Hugging Face *write* token: set "
            "HF_TOKEN, or run `hf auth login` (tokens: hf.co/settings/tokens)"
        )
    repo = strip_prefix(repo)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, exist_ok=True, private=True)

    ops = [CommitOperationAdd(path_in_repo="tokenizer.json", path_or_fileobj=path)]
    if not api.file_exists(repo_id=repo, filename="README.md"):
        lines = ["---", "library_name: tokenizers", "---", "",
                 f"Byte-level BPE tokenizer (`tokenizers` JSON format)."]
        for k, v in (details or {}).items():
            lines.append(f"- {k}: {v}")
        lines += ["", "```python",
                  "from tokenizers import Tokenizer",
                  f'tok = Tokenizer.from_pretrained("{repo}")',
                  "```"]
        card = "\n".join(lines) + "\n"
        ops.append(CommitOperationAdd(path_in_repo="README.md",
                                      path_or_fileobj=card.encode()))
    try:
        api.create_commit(repo_id=repo, operations=ops,
                          commit_message="upload tokenizer")
    except Exception as e:
        # create_repo(exist_ok=True) swallows the 403 when the repo id names
        # someone ELSE'S public repo, so no-write-access can first surface here.
        raise RuntimeError(
            f"upload to hf.co/{repo} failed: {e}\n"
            f"The token must have *write* scope and the namespace must be yours."
        ) from e
    print(f"pushed {path} -> hf.co/{repo} (private; tokenizer.json)")


# ----------------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------------


def _local_steps(save_dir: str) -> list[int]:
    if not os.path.isdir(save_dir):
        return []
    return [int(d) for d in os.listdir(save_dir) if d.isdigit()]


def _repo_steps(api, repo: str) -> list[int]:
    """Top-level numeric directories in the repo — the checkpoints it holds."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        files = api.list_repo_files(repo_id=repo)
    except RepositoryNotFoundError:
        return []
    return sorted({int(h) for h in (f.split("/", 1)[0] for f in files)
                   if h.isdigit()})


def hub_latest_step(repo: str) -> int | None:
    from huggingface_hub import HfApi

    steps = _repo_steps(HfApi(token=hf_token()), strip_prefix(repo))
    return max(steps) if steps else None


def push_checkpoint(save_dir: str, step: int, repo: str) -> None:
    """Upload save_dir/<step> plus config.json, replacing any older step.

    One create_commit both adds the new files and deletes the previous step
    directory, so the swap is atomic: the repo never holds a torn or stale
    checkpoint, and always exactly one — resume never has to guess. Local
    --keep still retains history for forking off the LR plateau.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    token = hf_token()
    if token is None:
        raise RuntimeError(
            "pushing checkpoints needs a Hugging Face *write* token: set "
            "HF_TOKEN, or run `hf auth login`"
        )
    repo = strip_prefix(repo)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, exist_ok=True, private=True)

    step_dir = os.path.join(save_dir, str(step))
    ops, total = [], 0
    for root, _, files in os.walk(step_dir):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, save_dir).replace(os.sep, "/")
            ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=full))
            total += os.path.getsize(full)
    if not ops:
        raise RuntimeError(f"nothing to push: {step_dir} is missing or empty")
    cfg = os.path.join(save_dir, "config.json")
    if os.path.isfile(cfg):
        ops.append(CommitOperationAdd(path_in_repo="config.json", path_or_fileobj=cfg))
    for old in _repo_steps(api, repo):
        if old != step:
            ops.append(CommitOperationDelete(path_in_repo=f"{old}/", is_folder=True))
    if not api.file_exists(repo_id=repo, filename="README.md"):
        card = "\n".join([
            "---", "library_name: flax", "---", "",
            "Orbax checkpoint of a Flax NNX MLM encoder, pushed by lm.py.",
            "Holds the newest step only: `<step>/` (params, Adam moments, "
            "data-stream position) plus `config.json`.", "",
            "Resume training with:", "```bash",
            f"python lm.py ... --hub-checkpoints {repo} --resume", "```",
        ]) + "\n"
        ops.append(CommitOperationAdd(path_in_repo="README.md",
                                      path_or_fileobj=card.encode()))
    t0 = time.time()
    try:
        api.create_commit(repo_id=repo, operations=ops,
                          commit_message=f"checkpoint step {step}")
    except Exception as e:
        raise RuntimeError(
            f"checkpoint upload to hf.co/{repo} failed: {e}\n"
            f"The token must have *write* scope and the namespace must be yours."
        ) from e
    try:
        # Deleting the old step only drops it from the repo *tree*; the LFS
        # blobs keep counting against the account quota (100 GB on the free
        # tier) until history is squashed — ~400 MB per push, forever. This
        # repo is a machine-managed mirror whose contract is "newest
        # checkpoint only", so its history has no value.
        api.super_squash_history(repo_id=repo)
    except Exception as e:
        print(f"  note: history squash failed ({e}); Hub storage grows by "
              f"~one checkpoint per push until it succeeds")
    print(f"pushed checkpoint step {step} ({total / 1e6:.0f} MB) -> "
          f"hf.co/{repo} in {time.time() - t0:.0f}s")


def pull_checkpoint(save_dir: str, repo: str) -> int:
    """Download the repo's checkpoint into save_dir when it is ahead of local.

    Returns the step that is now newest locally. Local wins ties, so an
    uninterrupted local run never re-downloads; when neither side has a
    checkpoint this raises rather than letting --resume die later on a bare
    FileNotFoundError for config.json.

    Crash-safe: the ~400 MB download lands in a staging directory (non-numeric,
    so neither _local_steps nor orbax can mistake it for a checkpoint) and the
    step directory is renamed into place only once it is complete. A pull
    killed mid-transfer therefore leaves nothing that could win the tie-break
    and wedge every later resume; a retry resumes the staging download
    (complete files are skipped by their hash).
    """
    from huggingface_hub import snapshot_download

    repo = strip_prefix(repo)
    hub_step = hub_latest_step(repo)
    local = _local_steps(save_dir)
    local_step = max(local) if local else None
    if hub_step is None and local_step is None:
        raise RuntimeError(
            f"no checkpoint on hf.co/{repo} or in {save_dir} yet — this looks "
            f"like the first session, so run the same command without --resume"
        )
    if hub_step is None or (local_step is not None and local_step >= hub_step):
        return local_step
    print(f"pulling checkpoint step {hub_step} from hf.co/{repo} ...", flush=True)
    staging = os.path.join(save_dir, f".pull-{hub_step}")
    snapshot_download(repo_id=repo, token=hf_token(), local_dir=staging,
                      allow_patterns=[f"{hub_step}/*", f"{hub_step}/**",
                                      "config.json"])
    cfg = os.path.join(staging, "config.json")
    if os.path.isfile(cfg):
        os.replace(cfg, os.path.join(save_dir, "config.json"))
    os.replace(os.path.join(staging, str(hub_step)),
               os.path.join(save_dir, str(hub_step)))
    shutil.rmtree(staging, ignore_errors=True)  # hf's .cache metadata remnants
    return hub_step
