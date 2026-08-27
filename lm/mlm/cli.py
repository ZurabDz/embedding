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
    p.add_argument("--tokenizer-path", default="ka_bpe.json")
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
                   help="continue the run in --save-dir, Adam moments included")
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

    if args.tokenize_to:
        from mlm.data import tokenize_to

        tokenize_to(args, args.tokenize_to)
        return

    from mlm.train import train

    train(args)
