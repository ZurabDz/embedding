# Distilling a Georgian embedding model (JAX / Flax NNX)

Train a sentence-embedding model that reproduces the geometry of a large
teacher (Google's `gemini-embedding-2`) on Georgian text. This is *knowledge
distillation for embeddings*, with two students sharing one training loop:

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
teacher            → artifacts/teacher_emb.npy (teacher vectors, cached)   [or: synthetic-teacher]
train              → <out-dir>/student_*.{json,msgpack} + teacher_mean.npy
eval               → correlation + retrieval demo
```

## Install

From the repo root (a [uv](https://docs.astral.sh/uv/) workspace — this
installs `geo-distill` and its `ka-mlm` dependency editable):

```bash
uv sync
```

On Kaggle/Colab: `!pip install -q -e ./lm -e ./geo_distill` from the repo root
(jax with the right accelerator wheel is preinstalled there; the packages only
add floors on top of it).

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
  one to watch; `train` keeps the checkpoint with the best val Spearman.
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
| `model.py` | tiny from-scratch encoder → mean-pool → projection → L2-norm |
| `mlm_student.py` | student wrapping the pretrained MLM encoder (head stripped) |
| `students.py` | the scratch/mlm registry + the typed `StudentConfig` |
| `losses.py` | cosine regression + the two similarity-matching losses |
| `metrics.py` | Pearson/Spearman similarity agreement, NN@1 |
| `checkpoint.py` | atomic writes; student config/params save + load |
| `train.py` | the distillation training loop |
| `eval.py` | correlation with teacher + nearest-neighbour retrieval demo |
