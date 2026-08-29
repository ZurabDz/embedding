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

Dual-GPU throughput
  The fp16 8B does not fit one T4, so the transformers backend splits the
  layers across both via device_map — a 2-stage pipeline. encode() packs
  length-sorted micro-batches under a token budget (--micro-batch-tokens)
  and keeps --pipeline-threads of them in flight; that overlap only
  materializes on GPUs with P2P, though: on Kaggle's PCIe T4s (no P2P) every
  cross-device tensor move serializes both GPUs, so they take turns and
  throughput caps at one-GPU-equivalent no matter what. `--backend vllm` is
  the real dual-GPU path there: tensor parallelism runs every layer on both
  GPUs at once (NCCL handles the no-P2P hop properly), with vLLM's own
  scheduler batching whole shards. The threaded transformers path stays safe
  because a plain Qwen3 forward with use_cache=False has no shared mutable
  state (per-forward RoPE, no KV cache, stateless non-offload hooks) — but
  only a plain forward: .generate() has real concurrency bugs and must never
  run through this model.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import queue
import threading
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
               instruction: str, corpus_hash: str,
               backend: str = "transformers") -> str:
    """Mirrors teacher.key() semantics: any setting that changes the vectors
    changes the key (and thereby the cache subdirectory). Backends differ by
    fp16 kernel noise (~0.999 cosine), so they key separately — but the
    transformers tag stays byte-identical to the pre-backend format, keeping
    every existing cache and pushed artifact seedable."""
    tag = f"{model}|{input_type}|{output_dim or 'full'}|{max_seq_len}|{instruction}"
    if backend != "transformers":
        tag += f"|{backend}"
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


def _plan_micro_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Split ascending `lengths` into contiguous index groups whose padded cost
    (group size × the group's longest length) stays within `budget` tokens.

    A single sequence longer than the whole budget still gets its own group —
    hard limits are the OOM path's job, not the planner's.
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    for i, n in enumerate(lengths):
        if cur and (len(cur) + 1) * n > budget:
            groups.append(cur)
            cur = []
        cur.append(i)
    if cur:
        groups.append(cur)
    return groups


def _clamp_budget(budget, cost: int, floor: int) -> bool:
    """An OOM at `cost` padded tokens proves that cost doesn't fit: clamp the
    (1-element, mutable) budget to cost // 2, never below floor, never up.

    Deliberately NOT "halve the budget per OOM": every in-flight group was
    planned at the same stale budget, so one overload episode would cascade
    the budget straight to the floor. Returns True if the budget shrank.
    """
    new = max(min(budget[0], cost // 2), floor)
    shrank = new < budget[0]
    budget[0] = new
    return shrank


def _assemble(parts, n: int) -> np.ndarray:
    """Scatter (group, rows) parts back to input order — refusing unfilled
    rows, so a worker that somehow died unrecorded surfaces as an error here
    and never as garbage embeddings in the shard cache."""
    got = sum(len(g) for g, _ in parts)
    if got != n:
        raise RuntimeError(
            f"encoder covered {got}/{n} rows — a pipeline worker died "
            f"without reporting; re-run (completed shards resume for free)")
    out = np.empty((n, parts[0][1].shape[1]), np.float32)
    for group, arr in parts:
        out[group] = arr
    return out


def _run_groups(groups, run_group, n_threads: int, on_oom):
    """Run index groups through run_group(group) -> array (one row per index).

    With n_threads > 1, daemon workers keep several groups in flight so both
    halves of a device_map-split model stay busy (GPU0 starts group i+1 while
    GPU1 finishes group i). An OOM-shaped failure splits the group in two and
    requeues it, then calls on_oom(group) — the encoder shrinks its token
    budget there; a single-index group that still OOMs becomes a SystemExit
    with sizing advice. Any other exception stops all workers and re-raises
    in the caller. Returns [(group, array), ...] in completion order.
    """
    todo: queue.Queue = queue.Queue()
    for g in groups:
        todo.put(g)
    results: list = []   # list.append is atomic under the GIL
    failures: list = []
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                group = todo.get_nowait()
            except queue.Empty:
                return
            try:
                results.append((group, run_group(group)))
            # BaseException: a SystemExit escaping a worker thread would die
            # silently and leave rows unfilled — trap everything, re-raise in
            # the caller. (No KeyboardInterrupt risk: signals hit the main
            # thread, and the sequential inline call re-raises it anyway.)
            except BaseException as exc:  # noqa: BLE001
                try:
                    if _is_oom(exc) and len(group) > 1:
                        mid = len(group) // 2   # requeue BEFORE the hook: a
                        todo.put(group[:mid])   # failing on_oom must not be
                        todo.put(group[mid:])   # able to drop these rows
                        on_oom(group)
                        continue
                    failures.append(SystemExit(
                        "GPU OOM on a single sentence — lower --max-seq-len "
                        "or --output-dim, or use a smaller --model "
                        "(Qwen3-Embedding-4B/0.6B)") if _is_oom(exc) else exc)
                except BaseException as hook_exc:  # noqa: BLE001
                    failures.append(hook_exc)  # even the handler failed:
                stop.set()                     # record it, stop everyone
                return

    if n_threads <= 1:
        worker()
    else:
        workers = [threading.Thread(target=worker, daemon=True)
                   for _ in range(n_threads)]
        for w in workers:
            w.start()
        try:
            for w in workers:
                while w.is_alive():   # join in slices so Ctrl-C still lands
                    w.join(0.2)
        except BaseException:         # KeyboardInterrupt: run() finalizes
            stop.set()
            raise
    if failures:
        raise failures[0]
    return results


def _balanced_device_map(model_id: str):
    """Equal-LAYER-count split of a qwen3 backbone across all visible GPUs.

    device_map="auto" balances by MEMORY, so GPU0 hosts the compute-free
    embed_tokens (1.2 GB on the 8B) plus fewer layers — the pipeline stages do
    unequal work and the lighter one idles. Returns None (caller keeps
    --device-map) for non-qwen3 models, <2 GPUs, or any lookup surprise.
    """
    try:
        import torch
        from transformers import AutoConfig

        n_gpus = torch.cuda.device_count()
        cfg = AutoConfig.from_pretrained(model_id)
        if n_gpus < 2 or getattr(cfg, "model_type", None) != "qwen3":
            return None
        per = -(-cfg.num_hidden_layers // n_gpus)  # ceil
        dm = {"embed_tokens": 0, "rotary_emb": 0, "norm": n_gpus - 1}
        for i in range(cfg.num_hidden_layers):
            dm[f"layers.{i}"] = min(i // per, n_gpus - 1)
        return dm
    except Exception:  # noqa: BLE001  (offline, exotic config, ...)
        return None


def make_encoder(args):
    """Build encode(texts) -> (n, D) float32, rows L2-normalized.

    The single seam between the pipeline logic and the model — tests replace
    this function, so ALL torch/transformers/vllm imports live below it
    (keeping `import geo_distill.local_teacher` as cheap as teacher.py's).
    """
    if getattr(args, "backend", "transformers") == "vllm":
        return _make_vllm_encoder(args)
    return _make_transformers_encoder(args)


def _make_vllm_encoder(args):
    """encode() via vLLM tensor parallelism: BOTH GPUs run every layer at
    once (NCCL copes with Kaggle's no-P2P PCIe, unlike the transformers
    split, whose stage handoffs serialize the two T4s into taking turns).

    vLLM is deliberately not a package extra — it pins its own torch — so it
    is installed per Kaggle session. The encoder owns its batching: run()
    hands it whole shards and vLLM's continuous-batching scheduler does the
    rest, which is why the micro-batch/pipeline flags don't apply here.
    """
    # Both must be set before vllm is imported: Kaggle notebooks already have
    # CUDA initialized (fork would break), and the T4s have no P2P (silence
    # the probe's warning).
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("NCCL_IGNORE_DISABLED_P2P", "1")
    try:
        from vllm import LLM
    except ImportError as exc:
        raise SystemExit(
            "--backend vllm needs vLLM (not an extra — it pins its own "
            "torch): pip install vllm==0.28.0   "
            "(sm75/T4 fallback pin: vllm==0.23.0)") from exc
    import torch

    tp = args.tensor_parallel or torch.cuda.device_count() or 1
    kwargs = dict(model=args.model, tensor_parallel_size=tp,
                  dtype="float16",          # T4s (sm_75) have no bfloat16
                  max_model_len=args.max_seq_len,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  enforce_eager=True)       # skip CUDA-graph capture: saves
    try:                                    # ~1 GiB + minutes on T4s
        llm = LLM(runner="pooling", **kwargs)
    except TypeError:  # pre-0.24 pins (the sm75 FlashInfer line) say task=
        llm = LLM(task="embed", **kwargs)
    try:
        from vllm import PoolingParams

        # Engine-side truncation for over-long inputs. NOTE: this keeps the
        # LAST max_seq_len tokens where the HF path keeps the FIRST — moot
        # for the <=300-char corpus, and the backends never share cache keys.
        pooling = PoolingParams(truncate_prompt_tokens=-1)
    except Exception:  # noqa: BLE001  (older PoolingParams without the field)
        pooling = None
    print(f"loaded {args.model} via vLLM (tensor_parallel={tp}, fp16, "
          f"max_len {args.max_seq_len})")

    def encode(texts):
        try:
            outs = llm.embed(list(texts), use_tqdm=False,
                             pooling_params=pooling)
        except TypeError:  # embed() without the pooling_params kwarg
            outs = llm.embed(list(texts), use_tqdm=False)
        emb = np.asarray([o.outputs.embedding for o in outs], dtype=np.float32)
        # vLLM's default pooler for this model already does last-token + L2
        # normalize (matching the HF reference); renormalizing is a no-op
        # guard, and Matryoshka stays here — never PoolingParams(dimensions=),
        # which vLLM rejects for repos that don't declare is_matryoshka.
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        if args.output_dim:
            if args.output_dim > emb.shape[1]:
                raise SystemExit(f"--output-dim {args.output_dim} exceeds the "
                                 f"model's native dim {emb.shape[1]}")
            emb = emb[:, :args.output_dim]
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return emb

    encode.owns_batching = True  # run() hands over whole shards
    return encode


def _make_transformers_encoder(args):
    """encode() via transformers: the fp16 8B splits across GPUs (device_map)
    as a 2-stage pipeline. One up-front tokenize per chunk, length-sorted
    micro-batches under --micro-batch-tokens, --pipeline-threads in flight
    (overlap needs GPU P2P — on Kaggle T4s the stages take turns regardless;
    use --backend vllm there). use_cache=False throughout — embedding never
    decodes, and the KV cache would cost ~144 KB/token on the 8B for nothing.
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

    device_map = args.device_map
    if args.balanced_split:
        device_map = _balanced_device_map(args.model) or args.device_map

    def load(dm):
        return AutoModel.from_pretrained(
            args.model, device_map=dm,
            # flash_attention_2 needs Ampere+; Kaggle T4s are Turing.
            attn_implementation="sdpa", **dtype_kw)

    try:
        model = load(device_map)
    except Exception:
        if device_map is args.device_map:
            raise
        print(f"  note: balanced split rejected by this model; falling back "
              f"to --device-map {args.device_map}")
        model = load(args.device_map)

    model.config.use_cache = False
    model.eval()
    if args.output_dim and args.output_dim > model.config.hidden_size:
        raise SystemExit(f"--output-dim {args.output_dim} exceeds the "
                         f"model's native dim {model.config.hidden_size}")
    n_params = sum(p.numel() for p in model.parameters())
    devices = sorted({str(p.device) for p in model.parameters()})
    print(f"loaded {args.model} ({n_params / 1e9:.1f}B params, "
          f"dtype {next(model.parameters()).dtype}, devices {devices}, "
          f"torch {torch.__version__}, transformers {transformers.__version__})")

    n_cuda = len({d for d in devices if d.startswith("cuda")})
    threads = args.pipeline_threads if n_cuda >= 2 else 1
    if threads > 1 and any(not d.startswith("cuda") for d in devices):
        # cpu/disk-offloaded layers use STATEFUL accelerate hooks (weights
        # move on every forward) — concurrent forwards are not safe there.
        print("  note: some layers are offloaded off-GPU; pipelining disabled")
        threads = 1
    budget = [max(args.micro_batch_tokens, args.max_seq_len)]
    lock = threading.Lock()
    if threads > 1:
        print(f"pipelining {threads} micro-batches (≤{budget[0]} tokens each) "
              f"across the {n_cuda}-GPU split")

    def encode(texts):
        enc = tok(texts, truncation=True, max_length=args.max_seq_len)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        # Plan against the PADDED width (pad_to_multiple_of=8 below), so the
        # budget matches what the forward actually allocates.
        plen = [-(-len(t) // 8) * 8 for t in ids]
        order = sorted(range(len(texts)), key=lambda i: plen[i])
        groups = [[order[j] for j in g] for g in
                  _plan_micro_batches([plen[i] for i in order], budget[0])]
        done = [0]

        def run_group(group):
            batch = tok.pad({"input_ids": [ids[i] for i in group],
                             "attention_mask": [mask[i] for i in group]},
                            pad_to_multiple_of=8,  # fp16 tensor-core alignment
                            return_tensors="pt").to(model.device)
            # inference_mode is thread-local: every worker opens its own.
            with torch.inference_mode():
                out = model(**batch, use_cache=False)
            emb = out.last_hidden_state[:, -1]  # left padding -> last pos = last real token
            emb = F.normalize(emb.float(), p=2, dim=1)
            if args.output_dim:
                # Matryoshka: truncate then re-normalize (no-op at the native
                # dim; the > native case was rejected at load time).
                emb = F.normalize(emb[:, :args.output_dim], p=2, dim=1)
            arr = emb.cpu().numpy()
            with lock:
                done[0] += len(group)
                print(f"    encoding {done[0]}/{len(texts)} of this batch",
                      end="\r", flush=True)
            return arr

        def on_oom(group):
            # Groups are ascending in length, so the group's last index is its
            # padded width; its cost is what the OOM actually proved too big.
            cost = len(group) * plen[group[-1]]
            with lock:
                shrank = _clamp_budget(budget, cost, args.max_seq_len)
            _cuda_empty_cache()
            if shrank:
                print(f"\n  GPU OOM; micro-batch budget -> {budget[0]} tokens")

        parts = _run_groups(groups, run_group, threads, on_oom)
        return _assemble(parts, len(texts))

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
        "backend": getattr(args, "backend", "transformers"),
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
    backend = getattr(args, "backend", "transformers")
    key = config_key(args.model, args.input_type, args.output_dim,
                     args.max_seq_len, instruction, chash, backend)
    store = ShardStore(args.cache_dir, key, len(sentences), args.checkpoint_every,
                       manifest_extra={"model": args.model,
                                       "input_type": args.input_type,
                                       "output_dim": args.output_dim,
                                       "max_seq_len": args.max_seq_len,
                                       "instruction": instruction,
                                       "backend": backend,
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
        # An encoder that owns its batching (vLLM) gets each shard whole and
        # schedules its own micro-batches.
        bs = (len(sentences) if getattr(encode, "owns_batching", False)
              else args.batch_size)
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
                          f"shard {i}/{len(missing)} · {rate:.1f}/s",
                          end="\r", flush=True)
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
