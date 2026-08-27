"""Command-line entry point: one subcommand per pipeline stage.

Flag parsing and dispatch only — validation happens up front, each stage
module's run(args) does the work, and heavy dependencies (datasets, google-genai,
jax) load lazily inside the stage that needs them so --help stays instant.
"""

import argparse
import sys

from geo_distill import config as paths


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="geo-distill",
        description="Distill Gemini teacher embeddings into a small Georgian "
                    "student model (see each subcommand's --help).")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="stream a HF corpus into clean, deduplicated "
                                    "Georgian sentences")
    d.add_argument("--dataset", default="ZurabDz/geo_small_corpus")
    d.add_argument("--split", default="train")
    d.add_argument("--n", type=int, default=20000, help="target number of sentences")
    d.add_argument("--min-chars", type=int, default=20)
    d.add_argument("--max-chars", type=int, default=300)
    d.add_argument("--out", default=paths.DATA_SENTENCES)
    d.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("tokenizer", help="train a BPE/Unigram tokenizer carrying "
                                         "the mlm special-token layout (ids 0..4)")
    t.add_argument("--input", default=paths.DATA_SENTENCES)
    t.add_argument("--out", default=paths.TOKENIZER_JSON)
    t.add_argument("--model", default="bpe", choices=["bpe", "unigram"])
    t.add_argument("--vocab-size", type=int, default=12000)
    t.add_argument("--min-frequency", type=int, default=2)

    te = sub.add_parser("teacher", help="embed the sentences with the Gemini "
                                        "teacher (cached, rate-limited; needs "
                                        "GEMINI_API_KEY)")
    te.add_argument("--input", default=paths.DATA_SENTENCES)
    te.add_argument("--cache", default=paths.TEACHER_CACHE)
    te.add_argument("--out-emb", default=paths.TEACHER_EMB)
    te.add_argument("--out-sents", default=paths.SENTENCES_JSON)
    te.add_argument("--out-meta", default=paths.TEACHER_META)
    te.add_argument("--usage", default=paths.TEACHER_USAGE)
    te.add_argument("--model", default=paths.DEFAULT_TEACHER_MODEL,
                    help="teacher model id — part of the embedding cache key, "
                         "so changing it re-embeds everything")
    te.add_argument("--batch-size", type=int, default=4)
    te.add_argument("--input-type", default="passage", choices=["passage", "query"],
                    help="Use one consistent type for the whole corpus.")
    te.add_argument("--output-dim", type=int, default=None,
                    help="Truncate the embedding (Matryoshka). Default: full.")
    te.add_argument("--rpm", type=int, default=100, help="requests per minute (0=off)")
    te.add_argument("--tpm", type=int, default=30000, help="tokens per minute (0=off)")
    te.add_argument("--rpd", type=int, default=100000,
                    help="requests per day (0=off); on the free tier pass 1000")
    te.add_argument("--chars-per-token", type=float, default=3.0,
                    help="local char->token ratio used only for the TPM budget")
    te.add_argument("--max-retries", type=int, default=6,
                    help="consecutive 429s tolerated before stopping gracefully")
    te.add_argument("--checkpoint-every", type=int, default=500,
                    help="snapshot the output files every N new embeddings so a "
                         "running train sees partial progress (0=only at end)")

    st = sub.add_parser("synthetic-teacher",
                        help="offline stand-in for the teacher: fabricates a "
                             "structured signal, no API key needed")
    st.add_argument("--tokenizer", default=paths.TOKENIZER_JSON)
    st.add_argument("--input", default=paths.DATA_SENTENCES)
    st.add_argument("--out-emb", default=paths.TEACHER_EMB)
    st.add_argument("--out-sents", default=paths.SENTENCES_JSON)
    st.add_argument("--dim", type=int, default=128)
    st.add_argument("--seed", type=int, default=0)

    tr = sub.add_parser("train", help="distill the teacher's geometry into a "
                                      "student (from scratch, or fine-tuning a "
                                      "pretrained MLM encoder)")
    tr.add_argument("--sentences", default=paths.SENTENCES_JSON)
    tr.add_argument("--teacher-emb", default=paths.TEACHER_EMB)
    tr.add_argument("--tokenizer", default=paths.TOKENIZER_JSON,
                    help="tokenizers-JSON path, or a Hub repo id (with "
                         "--mlm-checkpoint you want the tokenizer the encoder "
                         "was pretrained with, e.g. ZurabDz/ka-bpe-32k)")
    tr.add_argument("--out-dir", default=paths.ARTIFACTS_DIR)
    tr.add_argument("--mlm-checkpoint", default="",
                    help="initialise the student from a pretrained MLM encoder: "
                         "a Hub repo pushed by lm's --hub-checkpoints (e.g. "
                         "ZurabDz/ka-mlm) or a local lm --save-dir. The whole "
                         "encoder is fine-tuned; the from-scratch --dim/--depth/"
                         "--heads/--mlp-dim/--embed-dim flags are ignored")
    tr.add_argument("--dim", type=int, default=256)
    tr.add_argument("--depth", type=int, default=4)
    tr.add_argument("--heads", type=int, default=4)
    tr.add_argument("--mlp-dim", type=int, default=512)
    tr.add_argument("--max-len", type=int, default=64)
    tr.add_argument("--dropout", type=float, default=0.1)
    tr.add_argument("--embed-dim", type=int, default=None,
                    help="token-embedding width; < dim factorizes the table "
                         "(e.g. 128) and frees params for more depth")
    tr.add_argument("--epochs", type=int, default=120)
    tr.add_argument("--batch-size", type=int, default=512)
    tr.add_argument("--lr", type=float, default=None,
                    help="default 3e-4 from scratch, 5e-5 when fine-tuning a "
                         "pretrained encoder (--mlm-checkpoint)")
    tr.add_argument("--weight-decay", type=float, default=1e-2)
    tr.add_argument("--warmup", type=int, default=200)
    tr.add_argument("--val-frac", type=float, default=0.1)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--no-center", action="store_true",
                    help="do NOT mean-center the teacher targets. Centering "
                         "(the default) removes the shared component of the "
                         "teacher vectors; without it a from-scratch student "
                         "collapses onto the teacher centroid.")
    tr.add_argument("--reg-weight", type=float, default=1.0,
                    help="weight of the per-example cosine regression onto the "
                         "teacher vector (0 disables it)")
    tr.add_argument("--sim-weight", type=float, default=0.0,
                    help="weight of the within-batch similarity-matching aux "
                         "loss (0 = pure regression, the default)")
    tr.add_argument("--sim-loss", default="kl", choices=["mse", "kl"])
    tr.add_argument("--temperature", type=float, default=0.05)

    ev = sub.add_parser("eval", help="score the student against the teacher and "
                                     "demo retrieval")
    ev.add_argument("--model-dir", default=paths.ARTIFACTS_DIR,
                    help="directory holding student_config.json + "
                         "student_params.msgpack (train's --out-dir)")
    ev.add_argument("--sentences", default=paths.SENTENCES_JSON)
    ev.add_argument("--teacher-emb", default=paths.TEACHER_EMB)
    ev.add_argument("--val-frac", type=float, default=0.1)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--query", default=None)
    ev.add_argument("--topk", type=int, default=5)

    return p


def main(argv=None) -> None:
    p = build_parser()
    args = p.parse_args(argv)

    # Piped stdout is block-buffered by default, so plain prints would surface
    # long after the progress lines they belong between (very visible in
    # notebook cells, which capture both streams together).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    if args.cmd == "train" and args.reg_weight <= 0 and args.sim_weight <= 0:
        p.error("all loss weights are zero: raise --reg-weight or --sim-weight")

    if args.cmd == "data":
        from geo_distill.data import run
    elif args.cmd == "tokenizer":
        from geo_distill.tokenizer import run
    elif args.cmd == "teacher":
        from geo_distill.teacher import run
    elif args.cmd == "synthetic-teacher":
        from geo_distill.teacher import run_synthetic as run
    elif args.cmd == "train":
        from geo_distill.train import run
    else:
        from geo_distill.eval import run
    run(args)
