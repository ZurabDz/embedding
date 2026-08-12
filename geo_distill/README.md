# Distilling a tiny Georgian embedding model (JAX / Flax NNX)

Train a small, tailored sentence-embedding model that reproduces the **similarity
structure** of a large teacher (Google's `gemini-embedding-001`) on Georgian
text. This is *knowledge distillation for embeddings* — the point is educational:
see how a ~0.5–3M parameter model can inherit a big model's geometry on one domain.

## The core idea

The teacher outputs a high-dimensional vector per sentence. We don't try to copy
those vectors directly (the student can't even have the same dimensionality, and
matching raw coordinates is unnecessary). Instead we copy the **cosine-similarity
matrix**: for any batch of sentences, we want

```
cos(student_i, student_j)  ≈  cos(teacher_i, teacher_j)   for all i, j
```

This is *dimension-agnostic* — the student can output 128-dim vectors while the
teacher outputs 3072-dim — and it's exactly what you care about for retrieval,
clustering, and semantic search. Two loss options are provided:

- **`mse`** — mean-squared error between the two similarity matrices. Simple, stable.
- **`kl`** — treat each row's similarities (softmax with a temperature) as a
  *retrieval distribution* and minimize cross-entropy. Preserves nearest-neighbour
  ranking better; good when you mainly care about "which items are closest".

Because the teacher embeddings are **cached once**, training is fast, fully offline,
and costs no further API calls.

## Pipeline

```
data.py            → data/sentences.txt        (Georgian sentences)
train_tokenizer.py → artifacts/tokenizer.json  (small BPE, trained on your text)
embed_teacher.py   → artifacts/teacher_emb.npy  (teacher vectors, cached)
train.py           → artifacts/student_*.       (the distilled model)
eval.py            → correlation + retrieval demo
```

## Install

The project uses [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`):

```bash
uv sync
```

(`data.py` additionally needs `datasets` for the live HuggingFace download —
`uv add datasets` — but the offline quick-start below doesn't.) Or with pip:
`pip install -r requirements.txt`.

The student is written with **Flax NNX** — the current, PyTorch-like Flax API
where parameters live on the module instance — so it needs `flax>=0.12`.
GPU/TPU: the `jax[cuda12]` extra targets CUDA 12; swap it for the JAX wheel
matching your accelerator (see https://github.com/jax-ml/jax#installation).

## Run it

**1. Get data** (streams a slice, splits into clean sentences):
```bash
python data.py --n 20000
# or a bigger source:  python data.py --dataset RichNachos/georgian-corpus --n 50000
```

**2. Train the tokenizer** (tailored to your corpus — a big part of "small"):
```bash
python train_tokenizer.py --vocab_size 12000
```

**3. Cache teacher embeddings** (the only paid step; resumes if interrupted):
```bash
export GEMINI_API_KEY=AIza...          # get one at https://aistudio.google.com/apikey
python embed_teacher.py --batch_size 16 --input_type passage
# smaller/cheaper cache via Matryoshka truncation:  add  --output_dim 768
```
`embed_teacher.py` self-throttles to `--rpm`/`--tpm`/`--rpd` (defaulting to the
free tier: 100 req/min, 30K tok/min, 1000 req/day). If the daily request cap is
reached — or you Ctrl-C — it saves a *partial* `teacher_emb.npy` (with a
`teacher_meta.json` recording coverage) and resumes for free on the next run, so a
large corpus can be embedded across several days. Daily usage is tracked in
`teacher_usage.json`.

**4. Distill**:
```bash
python train.py --epochs 30 --batch_size 128 --loss mse
# ranking-focused variant:
python train.py --epochs 30 --batch_size 128 --loss kl --temperature 0.05
```

**5. Evaluate + try retrieval**:
```bash
python eval.py --query "საქართველოს ისტორია" --topk 5
```

### Try it offline first (no API key)

To exercise the whole thing before spending credits, a bundled Georgian sample
(`data/sentences.txt`, ~75 sentences) plus a synthetic teacher lets you run
steps 2→5 immediately:

```bash
python train_tokenizer.py --vocab_size 4000 --min_frequency 1
python make_synthetic_teacher.py
python train.py --epochs 40 --batch_size 16 --warmup 40
python eval.py --query "ქართული ღვინო და სუფრა"
```

`python sanity_test.py` overfits a single batch to confirm the model + gradients
are wired correctly.

## Reading the metrics

`eval.py` reports agreement between student and teacher on **held-out** sentences:

- **Pearson / Spearman** of the pairwise similarities — does the student place
  sentences in the same relative geometry? Spearman (rank correlation) is the one
  to watch; `train.py` keeps the checkpoint with the best val Spearman.
- **top-1 NN agreement** — for each sentence, is the student's nearest neighbour
  the same as the teacher's? This is the most retrieval-relevant number.

## Sizing & tips

- **Data.** A few thousand sentences is a starting point; tens of thousands is
  much better. Distillation quality is mostly bounded by how much (diverse) text
  the teacher gets to label. If val Spearman rises then falls while train loss
  keeps dropping, you're overfitting → get more data or add `--dropout 0.1`.
- **Student size.** Start `--dim 256 --depth 4 --heads 4 --out_dim 256`. Shrink
  for a faster/smaller model, grow if you're underfitting a large corpus.
- **Batch size matters for this loss.** Similarities are computed *within* each
  batch, so larger batches give a richer target matrix — use the biggest that fits.
- **`input_type`.** Embed the whole corpus with one consistent type (`passage`).
  A single symmetric student can't reproduce a query/passage-asymmetric teacher;
  keeping it consistent is the coherent choice for a first model. (`passage`/`query`
  map to Gemini's `RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY` task types.)

## Files

| file | what it does |
|------|--------------|
| `model.py` | tiny Flax NNX transformer encoder → mean-pool → projection → L2-norm |
| `distill_lib.py` | tokenization, the two distill losses, evaluation metrics |
| `data.py` | pull Georgian text from HuggingFace, split into sentences |
| `train_tokenizer.py` | train a small byte-level BPE tokenizer |
| `embed_teacher.py` | call the Gemini teacher, cache embeddings (resumable) |
| `train.py` | the distillation training loop |
| `eval.py` | correlation with teacher + nearest-neighbour retrieval demo |
| `make_synthetic_teacher.py` | offline stand-in for the teacher (no API key) |
| `sanity_test.py` | single-batch overfit check |
