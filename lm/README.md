# lm — MLM encoder pretraining (Flax NNX)

A small bidirectional MLM encoder pretrained on Georgian FineWeb-2, meant to be
fine-tuned afterwards (e.g. contrastively, for embeddings). Architecture follows
the ModernBERT/NeoBERT consensus: RoPE, pre-RMSNorm, SwiGLU, tied embeddings,
30% masking with 100% `[MASK]` replacement.

## Layout

```
lm.py            thin CLI entry point (python lm.py ... == python -m mlm ...)
mlm/
  config.py      token ids, corpus constants, EncoderConfig
  model.py       encoder, sublayers, losses
  masking.py     dynamic masking (grain RandomMap)
  data.py        corpus streaming, BPE tokenizer, windowing, split, grain pipeline
  optim.py       trapezoid LR schedule, AdamW
  sharding.py    single-host data parallelism, host->device streaming
  checkpoint.py  orbax: params + Adam moments + data-stream position
  train.py       jitted steps and the training loop
  selftest.py    architecture invariants asserted before a long run
  cli.py         flags and dispatch
```

## Quick start (no network needed)

```bash
python lm.py --selftest     # assert the architecture invariants
python lm.py --smoke        # tiny synthetic end-to-end run
```

## Real runs

```bash
pip install -r requirements.txt

# small corpus, tokenised in memory:
python lm.py --steps 20000 --docs 200000

# large corpus: tokenise once on CPU, then stream off disk while training —
# the GPU/TPU session spends none of its time on CPU work:
python lm.py --tokenize-to data/ka --docs 2000000
python lm.py --data-dir data/ka --steps 100000
```

Interrupted runs continue with their Adam moments *and* their exact position in
the data stream, so a resumed run matches an uninterrupted one:

```bash
python lm.py --data-dir data/ka --steps 100000 --resume
```

`--steps` is the *total* schedule length (it defines the LR trapezoid), not
"how many more steps" — keep it the same when resuming an unfinished run.

## Kaggle

The batch is sharded over every visible device automatically; `--batch-size`
must be divisible by the device count (2 on T4×2, 8 on TPU).

**Every session, first cell:**

```
!pip install -q flax optax orbax-checkpoint grain array_record datasets tokenizers
!git clone https://github.com/<you>/<repo>.git   # or add the code as a Kaggle dataset
%cd <repo>/lm
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

**Workflow across the ~9–12 h session cap:**

1. One CPU session: `!python lm.py --tokenize-to /kaggle/working/data_ka --docs 2000000`,
   then save the notebook output as a Kaggle dataset (this also produces
   `ka_bpe.json` — keep it, fine-tuning needs the same tokenizer).
2. Each training session: attach the tokenized dataset plus the *previous*
   session's checkpoint output, copy the checkpoints somewhere writable, and
   resume:

   ```
   !cp -r /kaggle/input/<prev-run>/checkpoints /kaggle/working/checkpoints
   !python lm.py --data-dir /kaggle/input/<tokenized-ds> --steps 100000 \
       --batch-size 256 --dtype bfloat16 \
       --save-dir /kaggle/working/checkpoints --resume
   ```

3. Save `/kaggle/working/checkpoints` as the session's output; repeat.

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
from mlm import load_checkpoint

model, cfg, step = load_checkpoint("checkpoints")   # inference-ready, eval mode
hidden = model.encode(input_ids, attention_mask)     # [B, L, hidden]
```

Mean-pool `hidden` over content tokens only (`input_ids >= N_SPECIAL`) — [CLS]
is an untrained attention sink here, and averaging it in folds a high-norm
outlier into every embedding. Expect the pooled space to be anisotropic until a
contrastive stage shapes it; MLM puts no loss on the aggregate of a sequence.
