"""Token ids, corpus constants, and the frozen model config."""

import dataclasses

import jax.numpy as jnp

PAD_ID, MASK_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]"]
N_SPECIAL = len(SPECIAL_TOKENS)

# Shortest window worth keeping. Below this a row is mostly [PAD] and carries
# too little context to predict from; above it, tails are worth their padding.
MIN_WINDOW_TOKENS = 32
# Held-out windows are sized in batches, not as a raw fraction: one eval batch
# makes the number pure noise.
MIN_VAL_BATCHES = 4


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    """Frozen so NNX can treat it as a static attribute of the modules."""

    vocab_size: int = 32_000
    hidden: int = 384
    layers: int = 12
    heads: int = 6
    mlp_hidden: int = 1024  # ~2.67x hidden, matches a 4x vanilla MLP's params
    max_len: int = 512
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0  # ModernBERT trains with 0.0; raise for tiny corpora
    init_std: float = 0.02
    # Compute and storage dtypes are separate on purpose: bf16 compute with fp32
    # master weights is the standard TPU recipe, and folding them into one field
    # would put Adam's moments in bf16 along with the parameters.
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    # Recompute every block's activations in the backward pass. Off by default;
    # this is what you reach for when you want a batch the card will not hold.
    remat: bool = False

    @property
    def head_dim(self) -> int:
        assert self.hidden % self.heads == 0
        return self.hidden // self.heads
