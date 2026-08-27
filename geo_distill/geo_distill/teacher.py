"""Embed sentences with the Gemini teacher model, respecting API rate limits.

This is the only step that costs API calls, so it caches aggressively AND paces
itself to stay inside your quota.

Caching
  Each embedding is written to artifacts/teacher_cache.jsonl keyed by a hash of
  (model, input_type, output_dim, text). Re-runs and interrupted runs resume for
  free, and switching teacher/settings never reuses a stale vector.

Rate limiting
  --rpm  requests per minute  — sliding-window throttle, sleeps as needed
  --tpm  tokens   per minute  — sliding-window throttle, tokens estimated locally
  --rpd  requests per day     — hard cap; usage is tracked in artifacts/teacher_usage.json
                                so several runs in one day share the same budget
                                (on the free tier pass --rpd 1000)
  When the daily cap is reached (or you Ctrl-C, or the server keeps returning 429),
  the script stops *gracefully*: it saves whatever it embedded so far and records
  metadata describing the partial coverage. Run it again (e.g. tomorrow, after the
  quota resets) to pick up where it left off — the cache makes that free.

Partial coverage
  If not everything fits in your quota, artifacts/{sentences.json, teacher_emb.npy}
  hold only the embedded subset (kept aligned 1:1 so train/eval just work), and
  artifacts/teacher_meta.json records how much of the input was covered.

The offline stand-in (`geo-distill synthetic-teacher`, run_synthetic below)
fabricates a *structured* teacher signal from bag-of-token counts, so the whole
pipeline runs without an API key or credits.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from geo_distill.checkpoint import atomic_np_save, atomic_write_text

# The passage/query "input_type" asymmetry maps onto Gemini task types. (Gemini
# offers others too, e.g. SEMANTIC_SIMILARITY / CLUSTERING — swap here to
# experiment.)
TASK_TYPES = {"passage": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


def key(text: str, input_type: str, model: str, output_dim) -> str:
    tag = f"{model}|{input_type}|{output_dim or 'full'}"
    return hashlib.sha1(f"{tag}\x00{text}".encode("utf-8")).hexdigest()


def load_cache(path: str) -> dict:
    cache = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    cache[rec["k"]] = rec["e"]
    return cache


def _l2norm(vec):
    arr = np.asarray(vec, dtype=np.float32)
    arr /= np.linalg.norm(arr) + 1e-12
    return arr.tolist()


def embed_batch(client, texts, model: str, input_type: str, output_dim):
    from google.genai import types

    cfg = types.EmbedContentConfig(task_type=TASK_TYPES[input_type])
    if output_dim:
        cfg.output_dimensionality = output_dim
    resp = client.models.embed_content(model=model, contents=list(texts), config=cfg)
    # Gemini returns embeddings in the same order as `contents`.
    embs = [e.values for e in resp.embeddings]
    # Full-dimensional output is already L2-normalized; truncated (Matryoshka)
    # output is not, so re-normalize. (Re-normalizing a unit vector is a no-op, so
    # this stays correct without hard-coding the model's native dimension.)
    if output_dim:
        embs = [_l2norm(e) for e in embs]
    return embs


def est_tokens(text: str, chars_per_token: float) -> int:
    """Cheap local token estimate for the TPM budget.

    We deliberately avoid count_tokens() so it doesn't burn requests. Gemini's
    tokenizer isn't public for Georgian, so this is approximate — the 429 backoff
    is the real safety net if we under-count.
    """
    return max(1, int(len(text) / chars_per_token) + 1)


def is_rate_limited(exc) -> bool:
    s = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return (code == 429 or "429" in s or "resource_exhausted" in s
            or "quota" in s or "rate limit" in s)


class RateLimiter:
    """Sliding-window limiter for requests/min + tokens/min, plus a hard requests/day cap.

    RPM/TPM are enforced by sleeping until the trailing-60s window has room. RPD is
    tracked on disk per calendar day, so multiple runs in one day share one budget.
    It's a courtesy guard to avoid hammering the API — the server's own 429s remain
    the real enforcement. A limit of 0 disables that dimension.
    """

    def __init__(self, rpm: int, tpm: int, rpd: int, usage_path: str):
        self.rpm, self.tpm, self.rpd = rpm, tpm, rpd
        self.usage_path = usage_path
        self.req_times: deque = deque()          # monotonic ts of recent requests
        self.tok_times: deque = deque()          # (monotonic ts, tokens) recent
        self.day, self.day_requests = self._load_usage()

    @staticmethod
    def _today() -> str:
        return datetime.date.today().isoformat()

    def _load_usage(self):
        try:
            with open(self.usage_path) as f:
                u = json.load(f)
            if u.get("date") == self._today():
                return u["date"], int(u.get("requests", 0))
        except Exception:  # noqa: BLE001  (missing/corrupt usage file -> start fresh)
            pass
        return self._today(), 0

    def _save_usage(self):
        try:
            with open(self.usage_path, "w") as f:
                json.dump({"date": self.day, "requests": self.day_requests}, f)
        except Exception:  # noqa: BLE001
            pass

    def daily_remaining(self) -> int:
        if self.day != self._today():            # midnight rollover -> reset
            self.day, self.day_requests = self._today(), 0
        return (1 << 30) if not self.rpd else max(0, self.rpd - self.day_requests)

    def acquire(self, tokens: int) -> bool:
        """Block until a request of `tokens` tokens fits RPM+TPM.

        Return False if the daily request cap is used up (caller should stop).
        """
        if self.daily_remaining() <= 0:
            return False
        while True:
            now = time.monotonic()
            while self.req_times and now - self.req_times[0] >= 60:
                self.req_times.popleft()
            while self.tok_times and now - self.tok_times[0][0] >= 60:
                self.tok_times.popleft()

            wait = 0.0
            if self.rpm and len(self.req_times) >= self.rpm:
                wait = max(wait, 60 - (now - self.req_times[0]))
            if self.tpm:
                cur = sum(t for _, t in self.tok_times)
                # Only wait if there's older token usage that will expire; a single
                # batch bigger than TPM can't be helped by waiting, so let it through
                # and let the server decide.
                if cur + tokens > self.tpm and self.tok_times:
                    wait = max(wait, 60 - (now - self.tok_times[0][0]))
            if wait <= 0:
                return True
            time.sleep(min(wait, 60) + 0.02)

    def record(self, tokens: int):
        now = time.monotonic()
        self.req_times.append(now)
        self.tok_times.append((now, tokens))
        self.day_requests += 1
        self._save_usage()


def finalize(sentences, cache, kfn, args) -> dict:
    """Write the embedded subset (aligned) + coverage metadata. Handles partial runs."""
    embedded = [s for s in sentences if kfn(s) in cache]
    if embedded:
        mat = np.array([cache[kfn(s)] for s in embedded], dtype=np.float32)
    else:
        mat = np.zeros((0, 0), dtype=np.float32)
    atomic_np_save(args.out_emb, mat)
    atomic_write_text(args.out_sents, json.dumps(embedded, ensure_ascii=False))

    dim = int(mat.shape[1]) if mat.ndim == 2 and mat.shape[1] else None
    meta = {
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
    }
    atomic_write_text(args.out_meta, json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def run(args) -> None:
    from google import genai
    from dotenv import load_dotenv

    # The key lives in the repo-root .dotenv (not the default ".env"), so point
    # python-dotenv at it explicitly — anchored to this file so it works from
    # any CWD. parents: [0]=geo_distill pkg, [1]=geo_distill project, [2]=repo.
    load_dotenv(Path(__file__).resolve().parents[2] / ".dotenv")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")
    client = genai.Client(api_key=api_key)

    with open(args.input, "r", encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    cache = load_cache(args.cache)

    def kfn(text):
        return key(text, args.input_type, args.model, args.output_dim)

    todo = [s for s in sentences if kfn(s) not in cache]
    limiter = RateLimiter(args.rpm, args.tpm, args.rpd, args.usage)
    print(f"{len(sentences)} sentences | {len(cache)} cached | {len(todo)} to embed "
          f"(model={args.model}, input_type={args.input_type})")
    print(f"limits: {args.rpm or '∞'} req/min · {args.tpm or '∞'} tok/min · "
          f"{args.rpd or '∞'} req/day  (daily budget remaining: {limiter.daily_remaining()})")

    cache_f = open(args.cache, "a", encoding="utf-8")
    i = 0
    bs = args.batch_size
    stop_reason = None
    consecutive_429 = 0
    since_ckpt = 0

    try:
        while i < len(todo):
            batch = todo[i : i + bs]
            tokens = sum(est_tokens(t, args.chars_per_token) for t in batch)

            if not limiter.acquire(tokens):
                stop_reason = "daily request cap reached — resume after the quota resets"
                break

            try:
                embs = embed_batch(client, batch, args.model, args.input_type,
                                   args.output_dim)
            except Exception as exc:  # noqa: BLE001
                if is_rate_limited(exc):
                    consecutive_429 += 1
                    if consecutive_429 > args.max_retries:
                        stop_reason = f"server kept rate-limiting after {args.max_retries} retries"
                        break
                    backoff = min(60, 2 ** consecutive_429)
                    print(f"\n  rate limited; backing off {backoff}s "
                          f"(retry {consecutive_429}/{args.max_retries})")
                    time.sleep(backoff)
                    continue
                # Non-quota error: usually a bad/oversized batch -> shrink and retry.
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"\n  error ({exc}); reducing batch size -> {bs}")
                    continue
                # A single item still failing: skip it so one bad line can't halt the run.
                print(f"\n  skipping unembeddable item: {batch[0][:60]!r} ({exc})")
                i += 1
                bs = args.batch_size
                continue

            consecutive_429 = 0
            limiter.record(tokens)
            for text, emb in zip(batch, embs):
                cache[kfn(text)] = emb
                cache_f.write(json.dumps({"k": kfn(text), "e": emb}) + "\n")
            cache_f.flush()
            i += len(batch)
            bs = args.batch_size  # recover to full batch size after any earlier shrink
            print(f"  embedded {i}/{len(todo)} this run · day {limiter.day_requests}"
                  f"/{args.rpd or '∞'} req", end="\r")

            since_ckpt += len(batch)
            if args.checkpoint_every and since_ckpt >= args.checkpoint_every:
                snap = finalize(sentences, cache, kfn, args)
                since_ckpt = 0
                print(f"\n  checkpoint: {snap['embedded']} embedded "
                      f"({snap['coverage'] * 100:.1f}%) written to {args.out_emb}")
    except KeyboardInterrupt:
        stop_reason = "interrupted (Ctrl-C)"
    finally:
        cache_f.close()

    meta = finalize(sentences, cache, kfn, args)
    print()
    if meta["complete"]:
        print(f"Embedded all {meta['embedded']} sentences | dim {meta['embedding_dim']}")
    else:
        print(f"Partial: {meta['embedded']}/{meta['input_total']} "
              f"({meta['coverage'] * 100:.1f}%) embedded"
              + (f" | dim {meta['embedding_dim']}" if meta['embedding_dim'] else ""))
        if stop_reason:
            print(f"   stopped: {stop_reason}")
        print("   Re-run to continue — the cache resumes already-embedded sentences for free.")
    print(f"   wrote {args.out_emb} · {args.out_sents} · {args.out_meta}")


def run_synthetic(args) -> None:
    """OFFLINE stand-in for the teacher endpoint, so the whole pipeline runs
    without spending any API credits. It fabricates a *structured* signal the
    student can actually learn: a random projection of L2-normalized
    bag-of-token counts, so sentences that share subwords get similar vectors.
    """
    from geo_distill.data import load_tokenizer

    tok = load_tokenizer(args.tokenizer)
    with open(args.input, encoding="utf-8") as f:
        sents = [ln.strip() for ln in f if ln.strip()]

    v = tok.get_vocab_size()
    bow = np.zeros((len(sents), v), dtype=np.float32)
    for i, enc in enumerate(tok.encode_batch(sents)):
        for t in enc.ids:
            bow[i, t] += 1.0
    bow /= (np.linalg.norm(bow, axis=1, keepdims=True) + 1e-8)

    rng = np.random.default_rng(args.seed)
    proj = rng.standard_normal((v, args.dim)).astype(np.float32) / np.sqrt(v)
    teacher = bow @ proj + 0.01 * rng.standard_normal(
        (len(sents), args.dim)).astype(np.float32)

    os.makedirs(os.path.dirname(args.out_emb) or ".", exist_ok=True)
    atomic_np_save(args.out_emb, teacher.astype(np.float32))
    atomic_write_text(args.out_sents, json.dumps(sents, ensure_ascii=False))
    print("synthetic teacher:", teacher.shape, "| sentences:", len(sents))
