# lm — MLM encoder pretraining (Flax NNX)

A small bidirectional MLM encoder pretrained on Georgian FineWeb-2, meant to be
fine-tuned afterwards (e.g. contrastively, for embeddings). Architecture follows
the ModernBERT/NeoBERT consensus: RoPE, pre-RMSNorm, SwiGLU, tied embeddings,
30% masking with 100% `[MASK]` replacement.

## Layout

```
lm.py            thin CLI shim (python lm.py ... == python -m mlm ... == mlm ...)
pyproject.toml   the installable `ka-mlm` package (workspace member; jax comes
                 from flax's own dependency, or the accelerator wheel already
                 on the host)
mlm/
  config.py      token ids, corpus constants, EncoderConfig (+ JSON codec)
  encoding.py    the wire format: [CLS] ... [SEP] rows, content-token pooling
  model.py       encoder, sublayers, losses
  masking.py     dynamic masking (pure numpy; the grain adapter lives in data.py)
  tokenizer.py   BPE training/resolution + the special-token contract
  data.py        corpus streaming, windowing, split, grain pipeline
  optim.py       trapezoid LR schedule, AdamW
  sharding.py    single-host data parallelism, host->device streaming
  checkpoint.py  orbax: params + Adam moments + data-stream position
  hub.py         Hugging Face Hub sync for tokenizers and checkpoints
  progress.py    terminal/notebook/log-adaptive progress bars
  train.py       jitted steps and the training loop
  selftest.py    architecture invariants asserted before a long run
  cli.py         flags and dispatch
```

`import mlm` needs only jax/flax/optax — grain, orbax, datasets and
huggingface_hub load lazily inside the functions that use them, so `--selftest`
and downstream fine-tuning (geo_distill) run on a lean install.

## Quick start (no network needed)

```bash
python lm.py --selftest     # assert the architecture invariants
python lm.py --smoke        # tiny synthetic end-to-end run
```

## Real runs

```bash
pip install -e .        # or `uv sync` at the repo root (installs the workspace)

# small corpus, tokenised in memory:
python lm.py --steps 20000 --docs 200000

# large corpus: tokenise once on CPU, then stream off disk while training —
# the GPU/TPU session spends none of its time on CPU work:
python lm.py --tokenize-to data/ka --docs 2000000
python lm.py --data-dir data/ka --steps 100000
```

### Tokenizer on the Hugging Face Hub

With a write token in the `HF_TOKEN` env var, a freshly trained (or
cache-reused) tokenizer can be pushed to the Hub — the repo is created
*private* on first push:

```bash
HF_TOKEN=hf_... python lm.py --tokenize-to data/ka --docs 2000000     --push-tokenizer-to <user>/ka-bpe-32k
```

`--tokenizer-path` then accepts a Hub id anywhere a local path works, so a
later session (or the fine-tuning stage) pulls the exact same vocabulary
without carrying the JSON around:

```bash
python lm.py --steps 20000 --docs 200000 --tokenizer-path <user>/ka-bpe-32k
# or explicitly: --tokenizer-path hf://<user>/ka-bpe-32k
```

Rules: an existing local file is reused when its vocab fits within
`--vocab-size` (and retrained in place when it exceeds it); a local directory
uses the `tokenizer.json` inside (e.g. a downloaded Hub repo); an
`hf://user/repo` or bare `user/repo` id downloads `tokenizer.json` from the
Hub (a typo'd id errors out — it is never silently retrained), except that a
bare id whose parent directory exists locally is treated as a path; any other
value trains a fresh BPE and saves it there. A tokenizer from the Hub or a
directory defines the vocabulary — `--vocab-size` is ignored, with a printed
note. Private repos need a token for loading too (`HF_TOKEN` or
`hf auth login`).

Interrupted runs continue with their Adam moments *and* their exact position in
the data stream, so a resumed run matches an uninterrupted one:

```bash
python lm.py --data-dir data/ka --steps 100000 --resume
```

### Checkpoints on the Hugging Face Hub

`--hub-checkpoints user/repo` syncs checkpoints with a (private) Hub repo in
both directions: pushed checkpoints atomically replace the repo's previous one
(the repo always holds exactly the newest, as `<step>/` + `config.json`,
mirroring `--save-dir`), and `--resume` pulls it back whenever it is newer
than anything local. The *same command* therefore chains across capped
sessions:

```bash
# every session, identical command (add --resume from the second one on):
HF_TOKEN=hf_... python lm.py --data-dir data/ka --steps 100000     --hub-checkpoints <user>/ka-mlm --push-every 2500 --resume
```

`--push-every 0` (the default) pushes only the final checkpoint; N > 0 (a
nonzero multiple of `--save-every`) also pushes mid-run so a killed session
loses at most N steps. Each push blocks training while it uploads (~a few
hundred MB), so pick N accordingly. A mid-run push failure only warns; the
final push fails loudly. Local `--keep` history is unaffected — the Hub holds
one checkpoint, the local dir keeps ten for forking off the LR plateau.

Hub storage note: deleting the old step only removes it from the repo *tree*;
the blobs would keep counting against your account quota (100 GB free tier)
— so after every push the repo history is squashed automatically, keeping
storage at ~one checkpoint. If the squash ever fails it warns and retries on
the next push. `--smoke` refuses `--hub-checkpoints` outright: a smoke run
never touches the Hub.

`--steps` is the *total* schedule length (it defines the LR trapezoid), not
"how many more steps" — keep it the same when resuming an unfinished run.

## Kaggle

The batch is sharded over every visible device automatically; `--batch-size`
must be divisible by the device count (2 on T4×2, 8 on TPU).

**Every session, first cell:**

```
!git clone https://github.com/<you>/<repo>.git   # or add the code as a Kaggle dataset
%cd <repo>
!pip install -q -e ./lm                          # jax stays the preinstalled CUDA/TPU build
%cd lm
!python -c "import jax; print(jax.devices())"    # confirm the accelerator is visible
```

**GPU (T4×2, P100).** Keep the default `--dtype float32` — T4 and P100 have no
bf16 units, so bfloat16 is emulated and slower. Use `--remat` when you want a
batch the card will not otherwise hold (~4% more compute for a large activation-
memory cut).

```
!python lm.py --data-dir /kaggle/input/<tokenized-ds> --steps 100000 \
    --batch-size 64 --remat --save-dir /kaggle/working/checkpoints
```

**TPU (8 devices).** Use bf16 compute (fp32 master weights) and a batch
divisible by 8:

```
!python lm.py --data-dir /kaggle/input/<tokenized-ds> --steps 100000 \
    --batch-size 256 --dtype bfloat16 --save-dir /kaggle/working/checkpoints
```

**Workflow across the ~9–12 h session cap.** Kaggle secrets are *not* env
vars — export the token in a notebook cell first, so `!python` subprocesses
inherit it (attach it via Add-ons → Secrets):

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

(Colab equivalent: `from google.colab import userdata;
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")`.) Then:

1. One CPU session: `!python lm.py --tokenize-to /kaggle/working/data_ka
   --docs 2000000 --push-tokenizer-to <user>/ka-bpe-32k`, then save the
   notebook output as a Kaggle dataset.
2. Every training session runs the *same* command — checkpoints round-trip
   through the Hub, so nothing is copied between sessions by hand:

   ```
   !python lm.py --data-dir /kaggle/input/<tokenized-ds> --steps 100000 \
       --batch-size 256 --dtype bfloat16 \
       --save-dir /kaggle/working/checkpoints \
       --hub-checkpoints <user>/ka-mlm --push-every 2500 --resume
   ```

   (Drop `--resume` only for the very first session; every later one pulls
   the newest checkpoint from the Hub automatically.)

## Progress reporting

Every long phase (corpus download, BPE training, encoding, training, eval)
reports progress, adapted to where the output lands:

- a real terminal: tqdm, redrawn in place;
- a Jupyter/Colab/Kaggle notebook kernel: tqdm's live notebook widget;
- `!python ...` cells, log files, nohup, CI: one plain line every ~10 s,
  greppable afterwards. Colab's `!` cells pretend to be a terminal (a pty)
  but cannot render tqdm's carriage-return redraws — they are detected via
  the COLAB_*/KAGGLE_* env vars and get plain lines too.

The first training step jit-compiles the whole graph (minutes on a GPU); a
notice says so, step 1 reports its own duration, and every rate/ETA after it
excludes the compile.

`--no-progress` switches to plain periodic prints everywhere.

## Fine-tuning

```python
from mlm import encode_sentences, load_checkpoint, pool_content

# dropout= overrides the pretraining config without touching the weights
model, cfg, step = load_checkpoint("checkpoints", dropout=0.1)
tokens, mask = encode_sentences(tokenizer, sentences, max_len=64)
hidden = model.encode(tokens, mask.astype("int32"))   # [B, L, hidden]
pooled = pool_content(hidden, tokens)                 # [B, hidden]
```

`pool_content` mean-pools over content tokens only (`ids >= N_SPECIAL`) — [CLS]
is an untrained attention sink here, and averaging it in folds a high-norm
outlier into every embedding. Expect the pooled space to be anisotropic until a
contrastive stage shapes it; MLM puts no loss on the aggregate of a sequence.
The packaged consumer of all this is `geo-distill train --mlm-checkpoint`
(see [`../geo_distill`](../geo_distill)), which fine-tunes the whole encoder
against a Gemini teacher.
