"""Self-test — the invariants worth asserting before a long run."""

import math

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from mlm.config import CLS_ID, MASK_ID, N_SPECIAL, PAD_ID, SEP_ID, EncoderConfig
from mlm.masking import mask_batch, n_predictions
from mlm.model import MlmEncoder, apply_rope, count_params, mlm_loss, rope_tables
from mlm.optim import make_optimizer, trapezoid_schedule
from mlm.sharding import to_device
from mlm.train import train_step


def selftest():
    cfg = EncoderConfig(vocab_size=500, hidden=64, layers=2, heads=4,
                        mlp_hidden=128, max_len=16)
    model = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    model.eval()
    rng = np.random.default_rng(0)
    ids = jnp.asarray(rng.integers(N_SPECIAL, 500, (1, 16)), jnp.int32)
    mask = jnp.ones((1, 16), jnp.int32)
    # the same sequence, but with positions 12..15 declared padding
    padded = jnp.concatenate(
        [jnp.ones((1, 12), jnp.int32), jnp.zeros((1, 4), jnp.int32)], axis=1
    )
    alt = ids.at[0, 12].set(77)

    def changed(attention_mask):
        a = model.encode(ids, attention_mask)
        b = model.encode(alt, attention_mask)
        return float(jnp.abs(a[0, 3] - b[0, 3]).max()) > 1e-6

    assert changed(mask), "position 3 ignores position 12 — attention is not bidirectional"
    assert not changed(padded), "position 12 is padded but still reaches position 3 — pad mask is broken"

    # `changed` only watches information travel backwards, from 12 to 3. An
    # anti-causal model — every position attending to later ones and nothing
    # else — satisfies it while being no more bidirectional than a GPT.
    base = model.encode(ids, mask)
    early = model.encode(ids.at[0, 1].set(88), mask)
    assert float(jnp.abs(base[0, 12] - early[0, 12]).max()) > 1e-6, \
        "position 12 ignores position 1 — attention only flows backward"

    # The RoPE asserts below test apply_rope in isolation, which says nothing
    # about whether the model calls it. Deleting both apply_rope lines from
    # BidirectionalAttention leaves every one of them passing and yields a bag
    # of words: permuting tokens then leaves the attended set unchanged.
    perm = ids.at[0, 3:6].set(ids[0, 3:6][::-1])
    assert float(jnp.abs(base[0, 0] - model.encode(perm, mask)[0, 0]).max()) > 1e-6, \
        "reordering tokens changed nothing — bag of words; RoPE is not wired into attention"

    # RoPE encodes *relative* position: with q and k held constant across
    # positions, the score q_i . k_j must depend only on the offset j - i, and
    # must actually vary with it. Both halves are load-bearing — a no-op
    # apply_rope is trivially "relative", and an absolute scheme does vary.
    cos, sin = rope_tables(16, 8, 10_000.0, jnp.float32)
    everywhere = lambda: jnp.broadcast_to(
        jnp.asarray(rng.standard_normal((1, 1, 1, 8)), jnp.float32), (1, 1, 16, 8)
    )
    qr = apply_rope(everywhere(), cos, sin)[0, 0]  # [16, 8]
    kr = apply_rope(everywhere(), cos, sin)[0, 0]
    scores = qr @ kr.T  # scores[i, j] = <rope(q, i), rope(k, j)>
    diags = [jnp.diagonal(scores, off) for off in range(-8, 9)]
    drift = max(float(d.max() - d.min()) for d in diags)
    spread = float(max(d.mean() for d in diags) - min(d.mean() for d in diags))
    assert drift < 1e-4, f"score varies within a fixed offset ({drift:.2e}) — RoPE is not relative"
    assert spread > 1e-2, f"score flat across offsets ({spread:.2e}) — apply_rope is a no-op"

    # a correctly initialised MLM head starts at ln(vocab_size)
    full = MlmEncoder(EncoderConfig(), rngs=nnx.Rngs(0))
    rows = rng.integers(N_SPECIAL, 32_000, (2, 128)).astype(np.int32)
    n_pred = n_predictions(128, 0.30)
    batch = {k: jnp.asarray(v)
             for k, v in mask_batch(rows, rng, 0.30, n_pred).items()}
    loss = float(mlm_loss(full, batch))
    # Bounded loosely on purpose. The expected value is ln(vocab) + sigma^2 / 2,
    # where sigma is the init logit spread, so the true centre sits just above
    # ln(vocab) and moves with the seed — a tight two-sided band flakes.
    assert abs(loss - math.log(32_000)) < 0.4, f"init loss {loss:.3f} != ln(vocab)"

    # mask_batch's contract, on a row that actually contains CLS/SEP/PAD — the
    # rows above are pure content, so none of this is otherwise exercised. A
    # mask_batch that forgot to mask at all passes every assert before this one.
    row = np.concatenate([[CLS_ID], rng.integers(N_SPECIAL, 500, 10), [SEP_ID], np.zeros(5, int)])
    rows_c = np.stack([row, row]).astype(np.int32)
    mb = mask_batch(rows_c, rng, 0.30, n_predictions(rows_c.shape[1], 0.30))
    pos, kept = mb["positions"], mb["weights"] > 0
    at_pos = np.take_along_axis(rows_c, pos, axis=1)
    assert (at_pos[kept] >= N_SPECIAL).all(), "a special or PAD token was chosen as a target"
    assert np.array_equal(mb["targets"][kept], at_pos[kept]), \
        "targets do not match the tokens at the gathered positions"
    assert (np.take_along_axis(mb["input_ids"], pos, axis=1)[kept] == MASK_ID).all(), \
        "a scored position was not replaced by [MASK] in the input"
    # ...and nothing else in the input moved
    expected = np.zeros_like(rows_c, dtype=bool)
    np.put_along_axis(expected, pos, kept, axis=1)
    assert np.array_equal(mb["input_ids"] != rows_c, expected), \
        "input_ids changed somewhere other than the scored positions"
    assert np.array_equal(mb["attention_mask"], (rows_c != PAD_ID).astype(np.int32)), "attention_mask is wrong"

    # The guard on M1: gathering the scored positions must not change the
    # objective. Scatter the targets back out and score every position the old
    # way; the two losses must agree.
    lab = np.full(rows.shape, -100, np.int32)
    np.put_along_axis(lab, np.asarray(batch["positions"]),
                      np.where(np.asarray(batch["weights"]) > 0,
                               np.asarray(batch["targets"]), -100), axis=1)
    lab = jnp.asarray(lab)
    dense = full(batch["input_ids"], batch["attention_mask"])  # [B, L, vocab]
    ce = optax.softmax_cross_entropy_with_integer_labels(
        dense.astype(jnp.float32), jnp.where(lab >= 0, lab, 0))
    wt = (lab >= 0).astype(jnp.float32)
    dense_loss = float((ce * wt).sum() / jnp.maximum(wt.sum(), 1.0))
    assert abs(dense_loss - loss) < 1e-4, \
        f"gathered loss {loss:.6f} != full-width loss {dense_loss:.6f} — the gather changed the objective"

    # The optimizer path is otherwise never executed here. A few steps on one
    # fixed batch must drive the loss down; if they do not, gradients are not
    # reaching the parameters and no amount of pretraining will help.
    small = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    small.train()
    opt = make_optimizer(small, trapezoid_schedule(1e-3, 40))
    fixed = to_device(mask_batch(rng.integers(N_SPECIAL, 500, (2, 16)).astype(np.int32),
                                 rng, 0.30, n_predictions(16, 0.30)), None)
    first = float(train_step(small, opt, fixed))
    for _ in range(39):
        last = float(train_step(small, opt, fixed))
    assert last < first - 0.1, f"40 steps on one batch moved the loss {first:.3f} -> {last:.3f}"

    # Same seed, same number — otherwise a run is not reproducible and two
    # evals are not comparable.
    twice = [float(mlm_loss(MlmEncoder(cfg, rngs=nnx.Rngs(7)), fixed)) for _ in range(2)]
    assert twice[0] == twice[1], f"same seed gave {twice[0]} then {twice[1]}"

    counts = count_params(full)
    print("self-test passed")
    print(f"  bidirectional both ways, position in use, pad mask, RoPE relativity")
    print(f"  mask contract, optimizer ({first:.2f} -> {last:.2f}), determinism, init loss {loss:.3f}")
    print(f"  default config: {counts['total'] / 1e6:.2f}M params "
          f"({counts['embedding'] / 1e6:.2f}M of it the tied table)")
