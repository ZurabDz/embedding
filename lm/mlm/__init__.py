"""A small bidirectional MLM encoder in Flax NNX — model, masking, and training
loop. Architecture follows the ModernBERT/NeoBERT consensus: RoPE, pre-RMSNorm,
SwiGLU, no biases except the tied decoder's, tied embeddings, 30% masking with
100% [MASK] replacement.

Quick start, no network needed:
    python lm.py --selftest
    python lm.py --smoke

Real run on Georgian FineWeb-2:
    pip install -e .          # the ka-mlm package (or `uv sync` at the repo root)
    python lm.py --steps 20000 --docs 200000

Corpora too large to hold in host RAM are tokenised once, up front, and then
streamed off disk — which also keeps a GPU session from spending its time on
CPU work:
    python lm.py --tokenize-to data/ka --docs 2000000
    python lm.py --data-dir data/ka --steps 100000

On TPU or a multi-GPU host the batch is sharded over every local device and
compute can drop to bf16 while the weights stay fp32:
    python lm.py --steps 20000 --docs 200000 --batch-size 128 --dtype bfloat16

Add --remat to trade ~4% more compute for a large cut in activation memory when
the batch you want will not otherwise fit.

Interrupted runs continue with their Adam moments *and* their exact position in
the data stream, so a resumed run matches an uninterrupted one:
    python lm.py --steps 20000 --resume

For downstream fine-tuning, the public surface is:
    from mlm import EncoderConfig, MlmEncoder, load_checkpoint
    from mlm import encode_sentences, pool_content, assert_special_tokens

Importing `mlm` deliberately needs only jax/flax/optax (+ numpy): the grain
input pipeline, orbax, datasets and huggingface_hub load lazily inside the
functions that use them, so a fine-tuning environment stays lean.
"""

from mlm.config import (CLS_ID, MASK_ID, N_SPECIAL, PAD_ID, SEP_ID,
                        SPECIAL_TOKENS, UNK_ID, EncoderConfig)
from mlm.encoding import cls_row, encode_sentences, pool_content
from mlm.model import MlmEncoder, count_params, mlm_loss
from mlm.checkpoint import load_checkpoint, resume_checkpoint
from mlm.tokenizer import assert_pad_is_zero, assert_special_tokens

__all__ = [
    "CLS_ID", "MASK_ID", "N_SPECIAL", "PAD_ID", "SEP_ID", "SPECIAL_TOKENS",
    "UNK_ID", "EncoderConfig", "MlmEncoder", "count_params", "mlm_loss",
    "load_checkpoint", "resume_checkpoint",
    "cls_row", "encode_sentences", "pool_content",
    "assert_pad_is_zero", "assert_special_tokens",
]
