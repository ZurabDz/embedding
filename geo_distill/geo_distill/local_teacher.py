"""Embed sentences with a local open-weights teacher (Qwen3-Embedding family).

Same job as teacher.py, minus the API: the model runs on whatever GPUs are
present (Kaggle T4x2 for the 8B default via device_map="auto"), so there are
no rate limits — the bottleneck is GPU throughput. The outputs are byte-for-
byte the same artifacts the Gemini teacher produces (teacher_emb.npy /
sentences.json / teacher_meta.json, aligned 1:1), so train/eval run unchanged.

Caching
  Instead of teacher.py's JSONL (which at Qwen3-8B's dims would be gigabytes
  of decimal text), embeddings land in fixed-boundary float16 .npy shards under
  artifacts/local_teacher_cache/<config-key>/. The config key hashes everything
  that changes the vectors (model, input type, output dim, max seq len,
  instruction, the corpus itself), so switching settings never reuses a stale
  shard — the same invalidation contract as teacher.key(). A killed run loses
  at most one shard (--checkpoint-every sentences).

Cross-session resume (Kaggle session caps)
  The artifacts themselves round-trip through a private HF dataset repo
  (--push-to, on by default; `geo-distill fetch-teacher` pulls them back).
  A fresh session that fetched partial artifacts re-seeds the shard cache from
  them (seed_store_from_outputs) and continues where the dead session stopped,
  mirroring how lm chains checkpoints across sessions.

Qwen3-Embedding specifics
  Left padding + last-token pooling; queries are instruction-prefixed while
  passages are embedded plain (the --input-type analog of Gemini task types);
  Matryoshka truncation + re-normalization for --output-dim; sdpa attention
  (flash-attention needs Ampere+, Kaggle T4s are Turing).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import time

import numpy as np

from geo_distill.checkpoint import atomic_np_save, atomic_write_text

DEFAULT_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query")


def resolve_instruction(args) -> str:
    """The instruction actually applied: "" for passages (never prefixed, never
    keyed), the flag or the Qwen3 default for queries."""
    if args.input_type != "query":
        return ""
    return args.instruction or DEFAULT_QUERY_INSTRUCTION


def format_text(text: str, input_type: str, instruction: str) -> str:
    # Qwen3-Embedding contract: documents plain, queries instruction-prefixed.
    if input_type == "query":
        return f"Instruct: {instruction}\nQuery:{text}"
    return text


def corpus_sha1(sentences: list[str]) -> str:
    return hashlib.sha1("\x00".join(sentences).encode("utf-8")).hexdigest()


def config_key(model: str, input_type: str, output_dim, max_seq_len: int,
               instruction: str, corpus_hash: str) -> str:
    """Mirrors teacher.key() semantics: any setting that changes the vectors
    changes the key (and thereby the cache subdirectory)."""
    tag = f"{model}|{input_type}|{output_dim or 'full'}|{max_seq_len}|{instruction}"
    return hashlib.sha1(f"{tag}\x00{corpus_hash}".encode("utf-8")).hexdigest()[:16]


class ShardStore:
    """Positional float16 shard cache under <root>/<config_key>/.

    Shard boundaries are fixed at [k*size, min((k+1)*size, n)); the file
    shard_<start>.npy exists iff that whole range is embedded, so resume is
    "skip the files that exist". manifest.json pins the config; on resume the
    manifest's shard_size wins over the flag so boundaries can never drift.
    """

    def __init__(self, root: str, key: str, n: int, shard_size: int,
                 manifest_extra: dict):
        self.key, self.n = key, n
        self.dir = os.path.join(root, key)
        os.makedirs(self.dir, exist_ok=True)
        if shard_size <= 0:
            shard_size = max(n, 1)  # 0 = one shard, i.e. only finalize at end
        mpath = os.path.join(self.dir, "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                m = json.load(f)
            # The key already hashes model/settings/corpus, so a mismatch here
            # means a hand-copied or corrupted directory — refuse loudly.
            if m.get("config_key") != key or m.get("n_sentences") != n:
                raise RuntimeError(
                    f"shard cache {self.dir} does not match this run "
                    f"(manifest key={m.get('config_key')}, n={m.get('n_sentences')} "
                    f"vs key={key}, n={n}) — delete the directory to start over")
            if int(m["shard_size"]) != shard_size:
                print(f"  note: resuming with the cache's shard size "
                      f"{m['shard_size']} (flag said {shard_size})")
            shard_size = int(m["shard_size"])
        else:
            manifest = {"config_key": key, "n_sentences": n,
                        "shard_size": shard_size,
                        "created": datetime.datetime.now().isoformat(timespec="seconds"),
                        **manifest_extra}
            atomic_write_text(mpath, json.dumps(manifest, ensure_ascii=False, indent=2))
        self.shard_size = shard_size

    def _path(self, start: int) -> str:
        return os.path.join(self.dir, f"shard_{start:08d}.npy")

    def ranges(self) -> list[tuple[int, int]]:
        s = self.shard_size
        return [(k * s, min((k + 1) * s, self.n))
                for k in range((self.n + s - 1) // s)]

    def has(self, start: int) -> bool:
        return os.path.isfile(self._path(start))

    def save(self, start: int, arr: np.ndarray) -> None:
        atomic_np_save(self._path(start), arr.astype(np.float16))

    def load(self, start: int) -> np.ndarray:
        return np.load(self._path(start))

    def complete_ranges(self) -> list[tuple[int, int]]:
        return [r for r in self.ranges() if self.has(r[0])]

    def embedded_count(self) -> int:
        return sum(e - s for s, e in self.complete_ranges())


def _is_oom(exc) -> bool:
    return ("out of memory" in str(exc).lower()
            or type(exc).__name__ == "OutOfMemoryError")


def _cuda_empty_cache() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001  (no torch / no CUDA -> nothing to free)
        pass


def make_encoder(args):
    """Build encode(texts) -> (n, D) float32, rows L2-normalized.

    The single seam between the pipeline logic and the model — tests replace
    this function, so ALL torch/transformers imports live here (keeping
    `import geo_distill.local_teacher` as cheap as teacher.py's import).
    """
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if args.dtype == "auto":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    # transformers 5.x renamed torch_dtype -> dtype; on 4.x the new name would
    # be *silently ignored* (an fp32 8B never fits the GPUs), so pick by version.
    dtype_kw = ({"dtype": dtype}
                if int(transformers.__version__.split(".")[0]) >= 5
                else {"torch_dtype": dtype})
    model = AutoModel.from_pretrained(
        args.model, device_map=args.device_map,
        # flash_attention_2 needs Ampere+; Kaggle T4s are Turing.
        attn_implementation="sdpa", **dtype_kw)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    devices = sorted({str(p.device) for p in model.parameters()})
    print(f"loaded {args.model} ({n_params / 1e9:.1f}B params, "
          f"dtype {next(model.parameters()).dtype}, devices {devices})")

    def encode(texts):
        batch = tok(texts, padding=True, truncation=True,
                    max_length=args.max_seq_len, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**batch)
        emb = out.last_hidden_state[:, -1]  # left padding -> last pos = last real token
        emb = F.normalize(emb.float(), p=2, dim=1)
        if args.output_dim:
            if args.output_dim > emb.shape[1]:
                raise SystemExit(f"--output-dim {args.output_dim} exceeds the "
                                 f"model's native dim {emb.shape[1]}")
            # Matryoshka: truncate then re-normalize (no-op at the native dim).
            emb = F.normalize(emb[:, :args.output_dim], p=2, dim=1)
        return emb.cpu().numpy()

    return encode


def finalize_shards(sentences, store: ShardStore, args, instruction: str) -> dict:
    """teacher.finalize's contract, fed from the shard cache: write the embedded
    subset (aligned 1:1, input order) + coverage metadata. Handles partial runs."""
    ranges = store.complete_ranges()
    embedded = [s for start, end in ranges for s in sentences[start:end]]
    if embedded:
        mat = np.vstack([store.load(start) for start, _ in ranges])
        mat = mat.astype(args.store_dtype)
    else:
        mat = np.zeros((0, 0), dtype=np.float32)
    atomic_np_save(args.out_emb, mat)
    atomic_write_text(args.out_sents, json.dumps(embedded, ensure_ascii=False))

    dim = int(mat.shape[1]) if mat.ndim == 2 and mat.shape[1] else None
    meta = {
        # teacher.finalize's keys, so consumers of the meta never care which
        # teacher produced it ...
        "model": args.model,
        "input_type": args.input_type,
        "output_dim_requested": args.output_dim,
        "embedding_dim": dim,
        "input_file": args.input,
        "input_total": len(sentences),
        "embedded": len(embedded),
        "coverage": round(len(embedded) / len(sentences), 4) if sentences else 0.0,
        "complete": len(embedded) == len(sentences),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        # ... plus local-teacher extras; config_key is what lets a fresh
        # session re-seed its shard cache from fetched artifacts.
        "config_key": store.key,
        "store_dtype": args.store_dtype,
        "max_seq_len": args.max_seq_len,
        "backend": "transformers",
        "device_map": args.device_map,
        "instruction": instruction,
    }
    atomic_write_text(args.out_meta, json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def seed_store_from_outputs(store: ShardStore, sentences, args) -> int:
    """Re-seed an empty shard cache from previously finalized artifacts.

    This is the cross-session resume path: a new Kaggle session has no local
    cache, but `fetch-teacher` restored the (possibly partial) artifacts of the
    dead session. When the meta's config_key matches this run exactly and the
    fetched sentences are a prefix of the corpus, every fully covered shard is
    rebuilt from the embedding matrix (float16 -> float32 -> float16 round-trips
    exactly, so seeded shards are byte-identical to the originals). Anything
    off — Gemini-shaped meta without a config_key, different settings, a
    different corpus — seeds nothing; re-embedding is always safe.
    """
    if store.embedded_count() > 0:
        return 0
    for path in (args.out_emb, args.out_sents, args.out_meta):
        if not os.path.isfile(path):
            return 0
    with open(args.out_meta, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("config_key") != store.key:
        return 0
    with open(args.out_sents, encoding="utf-8") as f:
        prev = json.load(f)
    emb = np.load(args.out_emb)
    if (emb.ndim != 2 or emb.shape[0] != len(prev)
            or prev != sentences[:len(prev)]):
        return 0
    seeded = 0
    for start, end in store.ranges():
        if end <= len(prev):  # a trailing partial shard is simply re-encoded
            store.save(start, emb[start:end])
            seeded += end - start
    return seeded


def run(args) -> None:
    push_repo = None if args.no_push else args.push_to
    if push_repo:
        # Fail fast on a read-scoped token BEFORE spending GPU hours (the same
        # up-front contract as mlm.hub.ensure_writable's docstring).
        from mlm import hub

        hub.ensure_writable(push_repo, repo_type="dataset")

    with open(args.input, "r", encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    instruction = resolve_instruction(args)
    chash = corpus_sha1(sentences)
    key = config_key(args.model, args.input_type, args.output_dim,
                     args.max_seq_len, instruction, chash)
    store = ShardStore(args.cache_dir, key, len(sentences), args.checkpoint_every,
                       manifest_extra={"model": args.model,
                                       "input_type": args.input_type,
                                       "output_dim": args.output_dim,
                                       "max_seq_len": args.max_seq_len,
                                       "instruction": instruction,
                                       "corpus_sha1": chash})

    seeded = seed_store_from_outputs(store, sentences, args)
    if seeded:
        print(f"seeded {seeded} embeddings from existing artifacts "
              f"(fetched from a previous session)")

    missing = [r for r in store.ranges() if not store.has(r[0])]
    todo = sum(e - s for s, e in missing)
    print(f"{len(sentences)} sentences | {store.embedded_count()} cached | "
          f"{todo} to embed (model={args.model}, input_type={args.input_type}, "
          f"key={key})")

    stop_reason = None
    if missing:
        encode = make_encoder(args)
        bs = args.batch_size
        done = 0
        t0 = time.monotonic()
        try:
            for i, (start, end) in enumerate(missing, 1):
                texts = [format_text(s, args.input_type, instruction)
                         for s in sentences[start:end]]
                # Length-sorted batches cut padding waste; results scatter back
                # to positional order (alignment is positional).
                order = sorted(range(len(texts)), key=lambda j: len(texts[j]))
                out = None
                j = 0
                while j < len(order):
                    idx = order[j:j + bs]
                    try:
                        embs = encode([texts[k] for k in idx])
                    except Exception as exc:  # noqa: BLE001
                        if _is_oom(exc):
                            if bs == 1:
                                raise SystemExit(
                                    "GPU OOM at batch size 1 — lower "
                                    "--max-seq-len or --output-dim, or use a "
                                    "smaller --model (Qwen3-Embedding-4B/0.6B)"
                                ) from exc
                            # Unlike API 429s, OOM is deterministic: halve for
                            # good, never grow back.
                            bs = max(1, bs // 2)
                            _cuda_empty_cache()
                            print(f"\n  GPU OOM; batch size -> {bs}")
                            continue
                        raise
                    if out is None:
                        out = np.empty((len(texts), embs.shape[1]), np.float32)
                    out[idx] = embs
                    j += len(idx)
                    done += len(idx)
                    rate = done / max(time.monotonic() - t0, 1e-9)
                    print(f"  embedded {done}/{todo} this run · "
                          f"shard {i}/{len(missing)} · {rate:.1f}/s", end="\r")
                store.save(start, out)  # whole shards only: a kill loses <= one
        except KeyboardInterrupt:
            stop_reason = "interrupted (Ctrl-C)"

    meta = finalize_shards(sentences, store, args, instruction)
    print()
    if meta["complete"]:
        print(f"Embedded all {meta['embedded']} sentences | dim {meta['embedding_dim']}")
    else:
        print(f"Partial: {meta['embedded']}/{meta['input_total']} "
              f"({meta['coverage'] * 100:.1f}%) embedded"
              + (f" | dim {meta['embedding_dim']}" if meta['embedding_dim'] else ""))
        if stop_reason:
            print(f"   stopped: {stop_reason}")
        print("   Re-run to continue — the shard cache resumes for free.")
    print(f"   wrote {args.out_emb} · {args.out_sents} · {args.out_meta}")

    if push_repo:
        from geo_distill.hub import push_teacher_data

        # Partial results are pushed too: the meta records coverage, and the
        # next session fetches, re-seeds its cache, finishes, and re-pushes.
        push_teacher_data(push_repo, args.out_emb, args.out_sents, args.out_meta)
