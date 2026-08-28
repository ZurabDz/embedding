# Georgian sentence embeddings

Two coupled projects, one uv workspace:

| package | directory | what it does |
| --- | --- | --- |
| `ka-mlm` (imports as `mlm`) | [`lm/`](lm/) | pretrains a small bidirectional MLM encoder on Georgian text (Flax NNX, RoPE/RMSNorm/SwiGLU) |
| `geo-distill` (imports as `geo_distill`) | [`geo_distill/`](geo_distill/) | distills teacher embeddings (Gemini API or a local Qwen3-Embedding model) into a student — either a tiny from-scratch model or the pretrained MLM encoder fine-tuned |

The two connect through Hugging Face Hub artifacts: `lm` pushes the encoder
([ZurabDz/ka-mlm](https://hf.co/ZurabDz/ka-mlm)) and its tokenizer
([ZurabDz/ka-bpe-32k](https://hf.co/ZurabDz/ka-bpe-32k)); `geo_distill`
consumes both via `--mlm-checkpoint` / `--tokenizer`. Teacher embeddings
generated on Kaggle GPUs round-trip through a private dataset repo the same
way (`local-teacher --push-to` / `fetch-teacher`).

## Install

Locally (uses [uv](https://docs.astral.sh/uv/); installs both packages editable
plus the dev group):

```bash
uv sync
uv run mlm --selftest            # sanity-check the encoder stack, no network
uv run pytest -q                 # offline test suite
```

On Kaggle/Colab (jax with the right accelerator wheel is preinstalled — the
packages only add floors on top of it):

```bash
%cd embedding
!pip install -q -e ./lm -e ./geo_distill
!python -m mlm --selftest
!python -m geo_distill --help
```

## Entry points

- `python -m mlm …` or the `mlm` script — pretraining (see [`lm/README.md`](lm/README.md)).
  `python lm.py …` from inside `lm/` still works.
- `python -m geo_distill <stage> …` or the `geo-distill` script — the
  distillation pipeline: `data | tokenizer | teacher | local-teacher |
  fetch-teacher | synthetic-teacher | train | eval` (see
  [`geo_distill/README.md`](geo_distill/README.md)).
