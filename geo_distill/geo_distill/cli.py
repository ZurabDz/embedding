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
        description="Distill teacher embeddings (the Gemini API, or a local "
                    "Qwen3-Embedding model) into a small Georgian student "
                    "model (see each subcommand's --help).")
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

    lt = sub.add_parser("local-teacher",
                        help="embed the sentences with a local open-weights "
                             "teacher (Qwen3-Embedding; needs torch+transformers "
                             "— preinstalled on Kaggle, or the [local-teacher] "
                             "extra locally)")
    lt.add_argument("--input", default=paths.DATA_SENTENCES)
    lt.add_argument("--cache-dir", default=paths.LOCAL_TEACHER_CACHE,
                    help="shard cache root; each settings combo gets its own "
                         "subdirectory, so nothing stale is ever reused")
    lt.add_argument("--out-emb", default=paths.TEACHER_EMB)
    lt.add_argument("--out-sents", default=paths.SENTENCES_JSON)
    lt.add_argument("--out-meta", default=paths.TEACHER_META)
    lt.add_argument("--model", default=paths.DEFAULT_LOCAL_TEACHER_MODEL,
                    help="HF model id — part of the cache config key, so "
                         "changing it re-embeds everything; Qwen3-Embedding-"
                         "0.6B/4B fit one GPU (or CPU for smoke runs)")
    lt.add_argument("--backend", default="transformers",
                    choices=["transformers", "vllm"],
                    help="vllm runs the model TENSOR-parallel — every layer "
                         "on all GPUs at once, the only true dual-GPU mode "
                         "on Kaggle's no-P2P T4s (install per session: pip "
                         "install vllm==0.28.0). transformers splits layers "
                         "across GPUs, which take turns. Backends differ by "
                         "fp16 kernel noise, so each gets its own cache key")
    lt.add_argument("--tensor-parallel", type=int, default=0,
                    help="vllm only: GPUs to shard across (0 = all visible)")
    lt.add_argument("--gpu-memory-utilization", type=float, default=0.88,
                    help="vllm only: fraction of each GPU vLLM may reserve")
    lt.add_argument("--batch-size", type=int, default=2048,
                    help="sentences handed to the encoder per call; the "
                         "encoder packs its own forward passes, so the "
                         "OOM-adaptive knob is --micro-batch-tokens "
                         "(transformers backend; vllm takes whole shards)")
    lt.add_argument("--micro-batch-tokens", type=int, default=16384,
                    help="transformers backend: padded-token budget per "
                         "forward pass; shrunk for good on GPU OOM (floor: "
                         "a single sequence)")
    lt.add_argument("--pipeline-threads", type=int, default=3,
                    help="transformers backend: micro-batches kept in flight "
                         "across a 2+ GPU split; overlap needs GPU P2P "
                         "(Kaggle T4s have none — use --backend vllm there); "
                         "1 = sequential (the single-device fallback too)")
    lt.add_argument("--balanced-split", action="store_true",
                    help="transformers backend: split by equal LAYER counts "
                         "instead of device_map=auto's memory balance "
                         "(qwen3 models on 2+ GPUs only; ignored otherwise)")
    lt.add_argument("--input-type", default="passage", choices=["passage", "query"],
                    help="Use one consistent type for the whole corpus.")
    lt.add_argument("--instruction", default=None,
                    help="instruction prefix for --input-type query (default: "
                         "a generic retrieval instruction); passages are "
                         "always embedded plain")
    lt.add_argument("--output-dim", type=int, default=1024,
                    help="Matryoshka truncation, 32..4096 for Qwen3-8B "
                         "(re-normalized; the native dim = no truncation). "
                         "The student's output head matches this.")
    lt.add_argument("--max-seq-len", type=int, default=512,
                    help="token truncation cap; batches pad to their own "
                         "longest sequence, so short batches never pay for this")
    lt.add_argument("--device-map", default="auto",
                    help='"auto" splits the model across all GPUs (the 8B '
                         'needs both Kaggle T4s); also "cpu", "cuda:0", ...')
    lt.add_argument("--dtype", default="auto",
                    choices=["auto", "float16", "bfloat16", "float32"],
                    help="model compute dtype; auto = float16 on CUDA, "
                         "float32 on CPU")
    lt.add_argument("--store-dtype", default="float32",
                    choices=["float32", "float16"],
                    help="dtype of the merged teacher_emb.npy (cache shards "
                         "are always float16); float16 halves the Hub upload")
    lt.add_argument("--checkpoint-every", type=int, default=2048,
                    help="cache shard size: resume granularity AND the most "
                         "work a crash can lose (0 = one shard, only at end)")
    lt.add_argument("--push-to", default=paths.DEFAULT_TEACHER_DATASET_REPO,
                    help="HF *dataset* repo for the artifacts, pushed after "
                         "generation (partial runs too); write access is "
                         "verified up front, before the GPU run")
    lt.add_argument("--no-push", action="store_true",
                    help="skip the Hub entirely (offline/smoke runs)")

    ft = sub.add_parser("fetch-teacher",
                        help="download teacher artifacts from a HF dataset "
                             "repo (pushed by local-teacher --push-to) into "
                             "artifacts/")
    ft.add_argument("repo", nargs="?", default=paths.DEFAULT_TEACHER_DATASET_REPO,
                    help="dataset repo id (default: %(default)s)")
    ft.add_argument("--out-dir", default=paths.ARTIFACTS_DIR)

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
    elif args.cmd == "local-teacher":
        from geo_distill.local_teacher import run
    elif args.cmd == "fetch-teacher":
        from geo_distill.hub import run_fetch as run
    elif args.cmd == "synthetic-teacher":
        from geo_distill.teacher import run_synthetic as run
    elif args.cmd == "train":
        from geo_distill.train import run
    else:
        from geo_distill.eval import run
    run(args)
