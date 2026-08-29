# Distilling a Georgian embedding model (JAX / Flax NNX)

Train a sentence-embedding model that reproduces the geometry of a large
teacher — Google's `gemini-embedding-2` via API, or a local open-weights
`Qwen/Qwen3-Embedding-8B` on free Kaggle GPUs — on Georgian text. This is
*knowledge distillation for embeddings*, with two students sharing one
training loop:

- a **tiny from-scratch transformer** (~0.5–3M params before the output
  projection) — the educational baseline;
- the **pretrained MLM encoder** from the sibling [`../lm`](../lm) project,
  fine-tuned whole (`--mlm-checkpoint`, ~34M params) — the recommended path:
  it already knows the language, so distillation only has to shape the space.

## The core idea

The *default* objective is per-example cosine regression: the student outputs
in the teacher's full space (`out_dim == teacher_dim`, set automatically) and
each embedding is pulled onto its own **mean-centered** teacher vector — a
full-information target per example. (Raw Gemini vectors share a large common
component; without centering a from-scratch student collapses onto the
centroid. The training-split mean is saved as `teacher_mean.npy` next to the
model, so the trained space stays reproducible: subtract it from a raw teacher
vector to map it into the student's space.)

Similarity-matrix matching can be layered on as an auxiliary loss via
`--sim-weight`:

- **`--sim-loss mse`** — mean-squared error between the two within-batch
  cosine-similarity matrices;
- **`--sim-loss kl`** — treat each row's similarities (softmax with a
  temperature) as a *retrieval distribution* and minimize cross-entropy;
  preserves nearest-neighbour ranking better.

Because the teacher embeddings are **cached once**, training is fast, fully
offline, and costs no further API calls.

## Pipeline

One CLI, one subcommand per stage (`python -m geo_distill <stage>` or the
installed `geo-distill` script; every stage has `--help`):

```
data               → data/sentences.txt        (Georgian sentences)
tokenizer          → artifacts/tokenizer.json  (BPE/Unigram, mlm-compatible specials)
teacher            → artifacts/teacher_emb.npy (teacher vectors, cached)
  [alternatives: local-teacher (GPU, no rate limits) · synthetic-teacher (offline)]
fetch-teacher      ← pulls the same artifacts back from a HF dataset repo
train              → <out-dir>/student_*.{json,msgpack} + teacher_mean.npy
                     + train_state.json   (resumable; --push-to syncs to a HF model repo)
fetch-student      ← pulls the trained student back from that model repo
eval               → correlation + retrieval demo
```

## Install

From the repo root (a [uv](https://docs.astral.sh/uv/) workspace — this
installs `geo-distill` and its `ka-mlm` dependency editable):

```bash
uv sync
# + torch/transformers for local CPU smoke runs of the local-teacher stage:
uv sync --extra local-teacher
```

On Kaggle/Colab: `!pip install -q -e ./lm -e ./geo_distill` from the repo root
(jax with the right accelerator wheel is preinstalled there; the packages only
add floors on top of it). Do **not** install the `local-teacher` extra there —
torch/transformers/accelerate are preinstalled with the right CUDA wheels and
must not be replaced.

## Run it

Artifacts land relative to the current directory (`data/`, `artifacts/`), so
run the stages from wherever you want them — locally this directory, on Kaggle
`/kaggle/working`.

**1. Get data** (streams a slice, splits into clean sentences):
```bash
python -m geo_distill data --n 20000
# or a bigger source:  python -m geo_distill data --dataset RichNachos/georgian-corpus --n 50000
```

**2. Train the tokenizer** (tailored to your corpus — a big part of "small").
Both flavours (`--model bpe|unigram`) reserve the five `mlm` special tokens at
ids 0–4, so any tokenizer trained here also works with `--mlm-checkpoint`:
```bash
python -m geo_distill tokenizer --vocab-size 12000
```

**3. Cache teacher embeddings** (the only paid step; resumes if interrupted):
```bash
export GEMINI_API_KEY=AIza...          # get one at https://aistudio.google.com/apikey
python -m geo_distill teacher --batch-size 16 --input-type passage
# smaller/cheaper cache via Matryoshka truncation:  add  --output-dim 768
```
The teacher stage self-throttles to `--rpm`/`--tpm`/`--rpd` (defaults: 100
req/min, 30K tok/min, 100000 req/day — **on the free tier pass `--rpd 1000`**).
If the daily request cap is reached — or you Ctrl-C — it saves a *partial*
`teacher_emb.npy` (with `teacher_meta.json` recording coverage) and resumes for
free on the next run, so a large corpus can be embedded across several days.
Daily usage is tracked in `teacher_usage.json`. The teacher model id is a flag
(`--model`, default `gemini-embedding-2`) and is part of the cache key —
changing it re-embeds everything.

**3b. Local teacher (Qwen3-Embedding — free, needs GPUs).** The API-free
alternative: `local-teacher` embeds the corpus with an open-weights teacher
(default `Qwen/Qwen3-Embedding-8B`, Apache-2.0, Georgian among Qwen3's 119
languages) and writes the *same three artifacts*, so everything downstream is
unchanged. There are no rate limits — a 100k-sentence corpus is hours of free
Kaggle GPU time instead of weeks of API quota.

```bash
python -m geo_distill local-teacher --backend vllm  # both GPUs at once (Kaggle: see below)
python -m geo_distill local-teacher                 # transformers fallback, no extra installs
python -m geo_distill local-teacher --no-push       # offline / smoke runs
# smaller teachers for one GPU or CPU smoke tests:
python -m geo_distill local-teacher --model Qwen/Qwen3-Embedding-0.6B --no-push
```

Details worth knowing:

- **Dims.** The 8B is natively 4096-dim; the default `--output-dim 1024`
  truncates + re-normalizes (Matryoshka), keeping `teacher_emb.npy` ~4× smaller
  and the student's output head small. Pass `--output-dim 4096` for the full
  space.
- **Hardware.** fp16 8B weights (~16 GB) need both Kaggle T4s, and the two
  backends use them very differently. `--backend vllm` shards every layer
  across both GPUs (tensor parallelism), so both compute simultaneously —
  the only mode that truly uses 2×T4 at once, because Kaggle's T4s have no
  P2P and any cross-GPU handoff otherwise serializes them. The default
  `transformers` backend instead splits the *layers* (`--device-map auto`)
  into a 2-stage pipeline: the encoder packs length-sorted micro-batches
  under a token budget (`--micro-batch-tokens`, shrunk for good on OOM) and
  keeps `--pipeline-threads` in flight, but on no-P2P GPUs the stages still
  take turns — throughput caps at roughly one-GPU-equivalent (the pipelining
  only overlaps on P2P-capable multi-GPU boxes). `--balanced-split` evens
  the two stages; `--pipeline-threads 1` forces plain sequential inference.
- **vLLM specifics.** Install per session (`pip install vllm==0.28.0`;
  sm75/T4 fallback pin `vllm==0.23.0`) — it pins its own torch, which is why
  it is not a package extra. The engine runs fp16 (T4s have no bf16),
  `enforce_eager`, and does its own batching, so `--batch-size` /
  `--micro-batch-tokens` / `--pipeline-threads` don't apply. Backends differ
  by fp16 kernel noise (~0.999 cosine), so each keys its own shard cache —
  a run started on one backend re-embeds under the other, by design.
- **Cache & resume.** Embeddings land in fp16 shards under
  `artifacts/local_teacher_cache/<config-key>/` every `--checkpoint-every`
  sentences; a killed run loses at most one shard and re-runs resume for free.
  The config key hashes model/settings/corpus, so changed settings never reuse
  stale vectors.
- **Kaggle session caps.** `--push-to` (default `ZurabDz/geo-teacher-qwen3-8b`,
  created private) uploads the artifacts after every run, *partial ones too*;
  a fresh session runs `fetch-teacher`, and `local-teacher` re-seeds its shard
  cache from the fetched artifacts and continues. Write access is verified
  up front, before any GPU time is spent.
- **Prompting.** `--input-type passage` (the default) embeds text plain;
  `query` applies Qwen3's `Instruct: ...\nQuery:` prefix (`--instruction`
  overrides the instruction). As with the Gemini teacher: one consistent type
  for the whole corpus.

A Kaggle T4×2 notebook, first cell to last:

```
!git clone https://github.com/<you>/embedding.git
%cd embedding
!pip install -q -e ./lm -e ./geo_distill
!pip install -q vllm==0.28.0        # ~5-10 min; replaces Kaggle's torch — do it FIRST
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
%cd /kaggle/working
!python -m geo_distill data --n 100000
# resuming a session that died mid-run? pull its partial results first:
# !python -m geo_distill fetch-teacher
!python -m geo_distill local-teacher --backend vllm
```

First vLLM session, before committing the full run: watch a shard with
`!nvidia-smi dmon -s u -c 30` from a second cell — with tensor parallelism
both GPUs should sit high *simultaneously* (the transformers split alternates
them). Also worth a one-off parity check: embed the same ~100 sentences with
both backends (`--no-push`, separate `--cache-dir`s) and confirm per-row
cosine ≥ 0.999 between the two `teacher_emb.npy` files. If vLLM misbehaves on
the T4s, retry with `vllm==0.23.0`, or drop `--backend vllm` entirely — the
transformers path needs no extra installs and resumes its own cache.

The training session (any machine) then starts with:

```bash
python -m geo_distill fetch-teacher    # → artifacts/teacher_emb.npy + sentences.json
python -m geo_distill train --mlm-checkpoint ZurabDz/ka-mlm --tokenizer ZurabDz/ka-bpe-32k \
    --push-to <user>/ka-embed --push-every 2 [--resume]
```

That training session is itself capped, so it checkpoints and resumes the same
way — see [4c](#4c-chaining-a-distillation-across-capped-sessions) below.

**4. Distill** (from scratch):
```bash
python -m geo_distill train --epochs 30 --batch-size 128
# add the similarity-matching auxiliary term:
python -m geo_distill train --epochs 30 --batch-size 128 --sim-weight 0.5 --sim-loss kl
```

**4b. Distill from the pretrained MLM encoder (recommended):**
```bash
python -m geo_distill train --mlm-checkpoint ZurabDz/ka-mlm --tokenizer ZurabDz/ka-bpe-32k \
    --epochs 10 --batch-size 128 --dropout 0.1
```

`--mlm-checkpoint` takes the Hub repo pushed by lm's `--hub-checkpoints` (or a
local lm `--save-dir`); `--tokenizer` must be the tokenizer the encoder was
pretrained with — the special-token layout and vocab size are **hard-checked**,
so a mismatched tokenizer fails immediately instead of silently corrupting
every row. Private repos need `HF_TOKEN` (the Colab secrets vault and
`hf auth login` work too). Skip the `tokenizer` stage in this flow; `data` and
`teacher` are unchanged. The from-scratch sizing flags (`--dim/--depth/...`)
are ignored and the default LR drops to 5e-5 (fine-tuning). `eval` works on
either kind unchanged.

**4c. Chaining a distillation across capped sessions.** Ten epochs over a
million sentences outlive a free Kaggle session, so `train` checkpoints and
resumes the way `lm` does. `--push-to` names a private HF **model** repo (the
student's twin of the teacher's dataset repo — separate Hub namespaces), and
write access is verified up front, before the corpus load and the encoder
download:

```bash
# session 1 — checkpoint locally every epoch, mirror to the Hub every 2
python -m geo_distill train --mlm-checkpoint ZurabDz/ka-mlm --tokenizer ZurabDz/ka-bpe-32k \
    --epochs 10 --batch-size 128 --dropout 0.1 \
    --push-to <user>/ka-embed --push-every 2

# session 2 — the identical command plus --resume; the repo's checkpoint is
# pulled first when it is newer than anything local
python -m geo_distill train --mlm-checkpoint ZurabDz/ka-mlm --tokenizer ZurabDz/ka-bpe-32k \
    --epochs 10 --batch-size 128 --dropout 0.1 \
    --push-to <user>/ka-embed --push-every 2 --resume

# anywhere, later — download the finished student and score it
python -m geo_distill fetch-student <user>/ka-embed
python -m geo_distill eval --model-dir artifacts --query "საქართველოს ისტორია"
```

What is in a checkpoint, and why:

- `student_params.msgpack` is the **best** epoch by held-out Spearman;
  `student_state.msgpack` is the **latest** parameters plus the Adam moments,
  the dropout rng counters and the shuffle position. They diverge the moment an
  epoch fails to improve — `eval` always wants the first, a continuation always
  wants the second. Restoring only parameters would re-warm the learning rate
  from zero and replay epoch 0's dropout masks; with all of it, a resumed run
  reproduces the uninterrupted one exactly.
- `--save-every N` (default 1) sets the local checkpoint cadence; `--push-every
  N` (default 0 = final only) the Hub one, and must be a multiple of it. Each
  push blocks training while it uploads and carries the optimizer moments
  (~2× the parameters), which is the price of being resumable *from the Hub*
  rather than only from the local directory.
- A resume must be the **same run**: everything that defines the training math
  — architecture, tokenizer, objective weights, seed, batch size, dropout, and
  the corpus itself — is fingerprinted, and a mismatch names the field that
  moved instead of silently training something else. `--epochs` is the one
  exception: raising it extends the run and re-stretches the cosine decay,
  which is announced rather than hidden.
- Growing the teacher corpus between sessions is a **hard error**, not a
  warning: a longer corpus rescales the schedule, redefines an epoch, and moves
  the mean the regression targets are centered on, so the saved parameters
  would be chasing a target space that shifted under them. Start a fresh run on
  the bigger corpus.
- Neither side clobbers the other by accident. A fresh run (no `--resume`)
  refuses to push over a repo that already holds a run — unlike the local
  directory, the repo's copy is the only one. And a `--resume` whose `--push-to`
  names a *different* run refuses before pulling, because promoting the repo's
  files would overwrite the local run's best parameters.

**5. Evaluate + try retrieval**:
```bash
python -m geo_distill eval --query "საქართველოს ისტორია" --topk 5
```

### Try it offline first (no API key)

A committed Georgian sample (`../tests/data/sample_sentences.txt`, 100
sentences) plus the synthetic teacher exercise the whole thing before spending
credits:

```bash
python -m geo_distill tokenizer --input ../tests/data/sample_sentences.txt --vocab-size 1000 --min-frequency 1
python -m geo_distill synthetic-teacher --input ../tests/data/sample_sentences.txt
python -m geo_distill train --epochs 40 --batch-size 16 --warmup 40
python -m geo_distill eval --query "ქართული ღვინო და სუფრა"
```

## Reading the metrics

`eval` reports agreement between student and teacher on **held-out** sentences
(the split is a stable content hash, so it never moves between runs):

- **Pearson / Spearman** of the pairwise similarities — does the student place
  sentences in the same relative geometry? Spearman (rank correlation) is the
  one to watch; `train` keeps the parameters of the best-Spearman epoch in
  `student_params.msgpack` (the resume checkpoint beside it holds the latest
  ones, which is a different thing — see 4c).
- **top-1 NN agreement** — for each sentence, is the student's nearest
  neighbour the same as the teacher's? The most retrieval-relevant number.

## Sizing & tips

- **Data.** A few thousand sentences is a starting point; tens of thousands is
  much better. Distillation quality is mostly bounded by how much (diverse)
  text the teacher gets to label. If val Spearman rises then falls while train
  loss keeps dropping, you're overfitting → get more data or add `--dropout 0.1`.
- **Student size.** Start `--dim 256 --depth 4 --heads 4`. Shrink for a
  faster/smaller model, grow if you're underfitting a large corpus. The output
  width always equals the teacher dim (cosine regression requires it).
- **Batch size** only matters for the *auxiliary* similarity losses, which see
  the within-batch matrix; the default per-example regression is
  batch-independent, so pick whatever trains fastest.
- **`--input-type`.** Embed the whole corpus with one consistent type
  (`passage`). A single symmetric student can't reproduce a
  query/passage-asymmetric teacher. (`passage`/`query` map to Gemini's
  `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types.)

## Layout

| module | what it does |
|--------|--------------|
| `cli.py` | the subcommand parser + dispatch (this file owns every flag) |
| `config.py` | shared artifact paths + the default teacher model id |
| `data.py` | corpus download; tokenizer loading, batch encoding, stable val split |
| `tokenizer.py` | train a BPE (shared `mlm` recipe) or Unigram, five specials at 0–4 |
| `teacher.py` | Gemini teacher client (cached, rate-limited) + the synthetic stand-in |
| `local_teacher.py` | local open-weights teacher (Qwen3-Embedding): fp16 shard cache, multi-GPU inference, OOM backoff |
| `hub.py` | push/pull the teacher artifacts (private HF *dataset* repo) and the student (private HF *model* repo) |
| `model.py` | tiny from-scratch encoder → mean-pool → projection → L2-norm |
| `mlm_student.py` | student wrapping the pretrained MLM encoder (head stripped) |
| `students.py` | the scratch/mlm registry + the typed `StudentConfig` |
| `losses.py` | cosine regression + the two similarity-matching losses |
| `metrics.py` | Pearson/Spearman similarity agreement, NN@1 |
| `checkpoint.py` | atomic writes; student config/params save + load; the resume checkpoint (latest params + Adam moments + rng) |
| `train.py` | the distillation training loop: checkpointing, `--resume`, and the Hub push cadence |
| `eval.py` | correlation with teacher + nearest-neighbour retrieval demo |
