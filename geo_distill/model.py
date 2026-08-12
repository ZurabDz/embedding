"""Small Flax transformer encoder that produces sentence embeddings.

The student is deliberately tiny. Its job is not to be a great language model on
its own, but to reproduce the *similarity structure* of a much larger teacher
embedding model on a specific domain (here: Georgian text).

Written with Flax NNX (the current, PyTorch-like API): submodules and parameters
live as attributes on the module instance, created in ``__init__`` with an
``nnx.Rngs`` and used directly in ``__call__`` — no separate init/apply step.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def l2_normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0,
                 *, rngs: nnx.Rngs):
        # Pre-LayerNorm transformer block (more stable to train than post-LN).
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=heads,
            in_features=dim,
            qkv_features=dim,
            dropout_rate=dropout,
            decode=False,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x, mask=None, deterministic=True):
        h = self.norm1(x)
        h = self.attn(h, mask=mask, deterministic=deterministic)
        x = x + h

        h = self.norm2(x)
        h = self.fc1(h)
        h = jax.nn.gelu(h)
        h = self.dropout(h, deterministic=deterministic)
        h = self.fc2(h)
        x = x + h
        return x


class EmbeddingModel(nnx.Module):
    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 4,
                 heads: int = 4, mlp_dim: int = 512, max_len: int = 128,
                 out_dim: int = 256, dropout: float = 0.0,
                 embed_dim: int | None = None, *, rngs: nnx.Rngs):
        """
        vocab_size: tokenizer vocabulary size
        dim:        hidden width
        depth:      number of transformer blocks
        out_dim:    dimensionality of the final sentence embedding
        embed_dim:  token-embedding width; when smaller than dim, the lookup
                    table is factorized (vocab x embed_dim, then a linear map
                    up to dim), freeing parameters for the transformer stack.
                    None means embed_dim == dim (no factorization).
        """
        embed_dim = dim if embed_dim is None else embed_dim
        self.max_len = max_len
        self.tok_emb = nnx.Embed(vocab_size, embed_dim, rngs=rngs)
        self.embed_proj = (None if embed_dim == dim
                           else nnx.Linear(embed_dim, dim, rngs=rngs))
        # Learned positional embedding, initialised ~ N(0, 0.02).
        self.pos_emb = nnx.Param(
            jax.random.normal(rngs.params(), (1, max_len, dim)) * 0.02
        )
        self.blocks = nnx.List([
            TransformerBlock(dim, heads, mlp_dim, dropout, rngs=rngs)
            for _ in range(depth)
        ])
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.proj = nnx.Linear(dim, out_dim, rngs=rngs)

    def __call__(self, tokens, pad_mask, deterministic=True):
        """tokens: (B, L) int32. pad_mask: (B, L) with 1 for real tokens, 0 for pad."""
        L = tokens.shape[1]

        tok_emb = self.tok_emb(tokens)
        if self.embed_proj is not None:
            tok_emb = self.embed_proj(tok_emb)
        x = tok_emb + self.pos_emb[:, :L, :]

        # Build (B, 1, L, L) attention mask so padding tokens are never attended to.
        attn_mask = nnx.make_attention_mask(pad_mask, pad_mask)

        for block in self.blocks:
            x = block(x, mask=attn_mask, deterministic=deterministic)

        x = self.norm(x)

        # Masked mean pooling over the sequence -> one vector per input.
        m = pad_mask[:, :, None].astype(x.dtype)          # (B, L, 1)
        summed = (x * m).sum(axis=1)                       # (B, dim)
        counts = jnp.clip(m.sum(axis=1), min=1.0)          # (B, 1)
        pooled = summed / counts

        out = self.proj(pooled)                            # (B, out_dim)
        return l2_normalize(out)


def param_count(model: nnx.Module) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))))
