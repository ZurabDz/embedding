"""How efficiently does each tokenizer represent Georgian?

    uv run python bench/tokenizer_fertility.py
    uv run python bench/tokenizer_fertility.py --input data/sentences.txt --out bench/results-fertility.json

The paper claims a Georgian-only 32k vocabulary splits Georgian into far fewer
tokens than a 250k-500k multilingual one. This measures that instead of
asserting it, on the project's own corpus, with four numbers per tokenizer:

  fertility        subword tokens per whitespace word — the standard metric.
                   1.0 would mean every word is one token.
  tokens/sentence  what the encoder actually pays per input.
  chars/token      compression; higher is better.
  Georgian vocab   how many of the tokenizer's entries contain a single
                   Mkhedruli character at all. A 501k-entry multilingual
                   vocabulary is not 501k entries of Georgian, and the
                   difference is what the embedding table is spent on.

Needs only `tokenizers` (already a dependency) plus `huggingface_hub` to fetch
the comparators — the tokenizer JSON files are a few MB each, no model weights
and no GPU, so this runs on a laptop in well under a minute.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

# (hf repo, display name). Chosen to match the models the paper already
# compares against on SIB-200, plus the teacher's own tokenizer.
TOKENIZERS = [
    ("ZurabDz/ka-bpe-32k",           "ჩვენი ka-bpe-32k"),
    ("Qwen/Qwen3-Embedding-8B",      "Qwen3-8B (მასწავლებელი)"),
    ("bert-base-multilingual-cased", "mBERT"),
    ("xlm-roberta-base",             "XLM-R"),
    ("intfloat/multilingual-e5-base", "multilingual-E5"),
    ("BAAI/bge-m3",                  "bge-m3"),
    ("jinaai/jina-embeddings-v3",    "jina-v3"),
    ("sentence-transformers/LaBSE",  "LaBSE"),
]

MKHEDRULI = re.compile(r"[ა-ჿ]")


OUR_HIDDEN_CONFIG = "artifacts/mlm_checkpoint/ZurabDz__ka-mlm/config.json"

# One sentence carried through every tokenizer, for the qualitative panel:
# "Georgian language processing with artificial intelligence".
EXAMPLE = "ქართული ენის დამუშავება ხელოვნური ინტელექტით"


def load(repo: str):
    """-> (Tokenizer, hidden_size|None). hidden_size comes from the model's own
    config.json so the embedding-cost column is measured, not remembered."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    hidden = None
    try:
        cfg = json.loads(pathlib.Path(hf_hub_download(repo, "config.json")).read_text())
        hidden = cfg.get("hidden_size") or cfg.get("d_model") or cfg.get("n_embd")
    except Exception:
        # our own tokenizer repo carries no model config; take the encoder's
        # width from the checkpoint the paper actually used
        local = pathlib.Path(OUR_HIDDEN_CONFIG)
        if local.exists():
            hidden = json.loads(local.read_text()).get("hidden")
    return tok, hidden


def georgian_vocab_entries(tok) -> int:
    """How many vocabulary entries contain a Mkhedruli character.

    Must go through decode(): byte-level BPE (ours, Qwen3) stores its vocabulary
    in GPT-2 byte-encoded form, so matching the raw vocabulary strings reports
    zero Georgian for exactly the tokenizers that are most Georgian.
    """
    n = tok.get_vocab_size()
    pieces = tok.decode_batch([[i] for i in range(n)], skip_special_tokens=False)
    return sum(1 for p in pieces if MKHEDRULI.search(p))


def measure(tok, sentences, hidden):
    words = sum(len(s.split()) for s in sentences)
    chars = sum(len(s) for s in sentences)
    # add_special_tokens=False so tokenizers that append [CLS]/[SEP] are not
    # penalised for a constant that has nothing to do with Georgian
    n_tokens = sum(len(e.ids) for e in tok.encode_batch(sentences, add_special_tokens=False))

    vocab_size = tok.get_vocab_size()
    georgian = georgian_vocab_entries(tok)
    ex_ids = tok.encode(EXAMPLE, add_special_tokens=False).ids
    return {
        "vocab_size": vocab_size,
        "georgian_entries": georgian,
        "georgian_frac": georgian / vocab_size,
        "tokens": n_tokens,
        "fertility": n_tokens / words,
        "tokens_per_sentence": n_tokens / len(sentences),
        "chars_per_token": chars / n_tokens,
        "hidden_size": hidden,
        "embedding_params": (vocab_size * hidden) if hidden else None,
        "example_n_tokens": len(ex_ids),
        "example_pieces": [tok.decode([i]) for i in ex_ids],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/sentences.txt")
    p.add_argument("--n", type=int, default=0, help="sentences to use (0 = all)")
    p.add_argument("--out", default="bench/results-fertility.json")
    args = p.parse_args()

    src = pathlib.Path(args.input)
    if not src.exists():
        sys.exit(f"corpus not found: {src}\n"
                 f"(run `python -m geo_distill data` first, or pass --input)")
    sentences = [ln.strip() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.n:
        sentences = sentences[: args.n]
    words = sum(len(s.split()) for s in sentences)
    print(f"corpus: {len(sentences):,} sentences, {words:,} words, "
          f"{sum(len(s) for s in sentences):,} chars")
    print(f"example: {EXAMPLE}\n")

    results = {}
    for repo, name in TOKENIZERS:
        try:
            tok, hidden = load(repo)
        except Exception as e:
            print(f"  {name:18s} SKIPPED ({type(e).__name__})")
            continue
        results[name] = {"repo": repo, **measure(tok, sentences, hidden)}
        r = results[name]
        print(f"  {name:18s} fertility {r['fertility']:.3f}  "
              f"ex {r['example_n_tokens']:>3d}tok  "
              f"tok/sent {r['tokens_per_sentence']:5.1f}  "
              f"chars/tok {r['chars_per_token']:4.2f}  "
              f"vocab {r['vocab_size']:>7,}  "
              f"Georgian {r['georgian_entries']:>6,} ({r['georgian_frac']:.1%})")

    # headline ratios against the Georgian-only tokenizer
    base = results.get("ჩვენი ka-bpe-32k")
    if base:
        print("\nrelative to ka-bpe-32k:")
        for name, r in results.items():
            if name == "ჩვენი ka-bpe-32k":
                continue
            print(f"  {name:18s} {r['fertility'] / base['fertility']:.2f}x the tokens, "
                  f"{r['vocab_size'] / base['vocab_size']:.1f}x the vocabulary"
                  + (f", {r['embedding_params'] / base['embedding_params']:.1f}x the "
                     f"embedding parameters" if r['embedding_params'] and base['embedding_params'] else ""))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"corpus": {"path": str(src), "sentences": len(sentences),
                                          "words": words},
                               "example": EXAMPLE,
                               "tokenizers": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
