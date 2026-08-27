"""Command-line entry point: flag parsing and dispatch."""

import argparse
import sys

from mlm import progress


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pretrain a small bidirectional MLM encoder (RoPE, "
                    "pre-RMSNorm, SwiGLU, tied embeddings) in Flax NNX.")
    p.add_argument("--smoke", action="store_true", help="tiny synthetic run, no network")
    p.add_argument("--selftest", action="store_true", help="assert the architecture invariants and exit")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    p.add_argument("--config", default="kat_Geor")
    p.add_argument("--docs", type=int, default=20_000,
                   help="documents to keep (rows under 200 characters are "
                        "filtered before counting)")
    p.add_argument("--tokenizer-path", default="ka_bpe.json",
                   help="tokenizer to use: an existing local tokenizers-JSON "
                        "file or a directory holding tokenizer.json, an "
                        "hf://user/repo or bare user/repo Hub id (tokenizer.json "
                        "is downloaded, never retrained; a bare id whose parent "
                        "dir exists locally is treated as a path), or a "
                        "not-yet-existing path to train a fresh BPE into")
    p.add_argument("--push-tokenizer-to", default="",
                   help="Hub repo id (user/repo) to upload the trained "
                        "tokenizer to as tokenizer.json; needs a write token "
                        "in the HF_TOKEN env var, and creates the repo "
                        "*private* on first push")
    p.add_argument("--vocab-size", type=int, default=32_000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--peak-lr", type=float, default=6e-4)
    p.add_argument("--mask-prob", type=float, default=0.30)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="compute dtype; parameters and Adam moments stay float32. "
                        "Use bfloat16 on TPU and Ampere+ GPUs; keep float32 on "
                        "T4/P100, which have no bf16 units")
    p.add_argument("--val-frac", type=float, default=0.01,
                   help="fraction of *documents* held out of training for eval")
    p.add_argument("--eval-every", type=int, default=500,
                   help="held-out eval interval; 0 to evaluate only at the end")
    p.add_argument("--save-dir", default="checkpoints",
                   help="orbax checkpoint directory; pass an empty string to disable")
    p.add_argument("--save-every", type=int, default=500)
    # Keep enough that some survive on the plateau. At --save-every 500 a 20k
    # step run with --keep 3 retains only checkpoints inside the decay ramp,
    # which is exactly what the trapezoid schedule exists to let you fork from.
    p.add_argument("--keep", type=int, default=10, help="checkpoints to retain")
    p.add_argument("--resume", action="store_true",
                   help="continue the run in --save-dir, Adam moments included; "
                        "with --hub-checkpoints, the repo's checkpoint is pulled "
                        "first when it is newer than anything local")
    p.add_argument("--hub-checkpoints", default="",
                   help="Hub repo id (user/repo) to sync checkpoints with: every "
                        "pushed checkpoint atomically replaces the repo's previous "
                        "one (created private; needs a write token), and --resume "
                        "pulls it back — the same command then chains across "
                        "capped Kaggle/Colab sessions")
    p.add_argument("--push-every", type=int, default=0,
                   help="push to --hub-checkpoints every N steps (a multiple of "
                        "--save-every); 0 pushes only the final checkpoint. Each "
                        "push blocks training while it uploads")
    p.add_argument("--remat", action="store_true",
                   help="recompute block activations in the backward pass: large "
                        "memory saving for roughly 4%% more compute")
    p.add_argument("--tokenizer-docs", type=int, default=200_000,
                   help="documents used to fit the BPE; a 32k vocab is converged "
                        "long before this, and more only costs time")
    p.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True,
                   help="live progress bars; --no-progress restores plain periodic lines")
    p.add_argument("--tokenize-to", default="",
                   help="write train/val ArrayRecord files to this directory and exit")
    p.add_argument("--data-dir", default="",
                   help="train from ArrayRecord files written by --tokenize-to, "
                        "instead of building the corpus in memory")
    return p


def main(argv=None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    progress.PROGRESS = args.progress
    # Piped stdout is block-buffered by default, so plain prints would surface
    # long after the progress lines they belong between. Colab captures both
    # streams into one cell, which makes the reordering very visible.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if args.push_tokenizer_to:
        # Fail before any corpus work, not an hour into it.
        from mlm.hub import ensure_writable, hf_token

        if args.data_dir:
            p.error("--push-tokenizer-to does nothing with --data-dir: the "
                    "corpus is already tokenised; push during --tokenize-to "
                    "or an in-memory run instead")
        if hf_token() is None:
            p.error("--push-tokenizer-to needs a Hugging Face write token: "
                    "set HF_TOKEN, or run `hf auth login`")
        if not args.smoke:
            # One round-trip that also validates the token's *write* scope and
            # the namespace — presence alone lets a read-scoped Kaggle token
            # die at push time, hours in.
            try:
                ensure_writable(args.push_tokenizer_to)
            except RuntimeError as e:
                p.error(str(e))

    if args.selftest:
        from mlm.selftest import selftest

        selftest()
        return

    if args.smoke:
        # Only override what the caller left at its default, so a flag passed
        # explicitly alongside --smoke still means what it says.
        for name, value in [("docs", 200), ("seq_len", 128), ("steps", 30),
                            ("batch_size", 8), ("layers", 4), ("hidden", 128),
                            ("heads", 4), ("log_every", 5), ("eval_every", 10),
                            ("save_every", 15), ("val_frac", 0.05)]:
            if getattr(args, name) == p.get_default(name):
                setattr(args, name, value)

    if args.hub_checkpoints:
        from mlm.hub import ensure_writable, hf_token

        if args.smoke:
            # A --smoke run must not touch the network — and with every guard
            # (ensure_writable, the fresh-run hub check) smoke-disabled, its
            # final push would atomically REPLACE the repo's real checkpoint
            # with a 30-step toy one. Refuse the combination outright.
            p.error("--hub-checkpoints cannot be combined with --smoke: a "
                    "smoke run must not touch the Hub (its toy checkpoint "
                    "would replace the repo's real one)")
        if args.tokenize_to:
            p.error("--hub-checkpoints does nothing with --tokenize-to; it "
                    "syncs training checkpoints")
        if not args.save_dir:
            p.error("--hub-checkpoints needs --save-dir: checkpoints are "
                    "saved locally, then pushed")
        if args.push_every and (not args.save_every
                                or args.push_every % args.save_every):
            p.error(f"--push-every {args.push_every} needs --save-every to be "
                    f"a nonzero divisor of it (got {args.save_every}), or "
                    f"pushes never line up with a saved checkpoint")
        if hf_token() is None:
            p.error("--hub-checkpoints needs a Hugging Face write token: set "
                    "HF_TOKEN, or run `hf auth login`")
        try:
            ensure_writable(args.hub_checkpoints)
        except RuntimeError as e:
            p.error(str(e))

    if args.tokenize_to:
        from mlm.data import tokenize_to

        tokenize_to(args, args.tokenize_to)
        return

    from mlm.train import train

    train(args)
