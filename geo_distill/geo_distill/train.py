"""Distill the teacher's embedding geometry into the student.

Reads the cached teacher embeddings + the tokenizer, then trains the student
with per-example cosine regression: the student outputs directly in the
teacher's full embedding space (out_dim == teacher_dim) and each student
embedding is pulled onto its own teacher vector (LEAF-style, no PCA). The
targets are mean-centered first (raw Gemini vectors share a large common
component; without centering a from-scratch student collapses onto the
centroid). This gives every example a full-information target instead of only
the weak, batch-relative similarity signal.

Optionally (--sim-weight > 0) a within-batch similarity-matching term can be
added on top (KL over soft-retrieval distributions, or MSE on the Gram
matrices), but ablations show pure regression is enough, so it is off by default.
"""
from __future__ import annotations

import os

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx

from geo_distill import config as paths
from geo_distill.checkpoint import (atomic_np_save, save_student_config,
                                    save_student_params)
from geo_distill.data import load_tokenizer, val_split
from geo_distill.losses import cosine_regression_loss, kl_sim_loss, mse_sim_loss
from geo_distill.metrics import similarity_agreement
from geo_distill.model import EmbeddingModel, embed_in_batches, param_count
from geo_distill.students import StudentConfig, spec_for


def run(args) -> None:
    import json

    # ---- load data -------------------------------------------------------- #
    with open(args.sentences, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    teacher = np.load(args.teacher_emb).astype(np.float32)
    assert len(sentences) == teacher.shape[0], "sentences / teacher_emb misaligned"
    tok = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    print(f"{len(sentences)} pairs | teacher dim {teacher.shape[1]} | vocab {vocab_size}")

    spec = spec_for("mlm" if args.mlm_checkpoint else "scratch")
    # Hard contract check, not a warning: a tokenizer whose special ids don't
    # match the student's row layout silently corrupts every row.
    spec.validate_tokenizer(tok)
    tokens, mask = spec.encode(tok, sentences, args.max_len)

    # ---- train / val split (stable & content-based so runs stay comparable) #
    train_idx, val_idx = val_split(sentences, args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed)   # for per-epoch minibatch shuffling

    tr_tokens, tr_mask, tr_teacher = tokens[train_idx], mask[train_idx], teacher[train_idx]
    va_tokens, va_mask, va_teacher = tokens[val_idx], mask[val_idx], teacher[val_idx]

    # ---- regression targets: the teacher vectors, no dim projection -------- #
    # The student outputs in the teacher's full space (out_dim == teacher_dim).
    # We mean-center the targets (train mean only): raw Gemini vectors share a
    # large common component (mean pairwise cosine ~0.67), so without centering
    # a from-scratch student collapses onto that centroid and learns no
    # discriminative geometry. Centering spreads the targets over the sphere.
    out_dim = teacher.shape[1]
    center = not args.no_center
    if center:
        teacher_mean = tr_teacher.mean(axis=0, keepdims=True)
        centered = tr_teacher - teacher_mean
        tr_target = centered / (np.linalg.norm(centered, axis=-1, keepdims=True) + 1e-8)
        print(f"regression targets: mean-centered teacher {out_dim} dims")
    else:
        tr_target = tr_teacher
        print(f"regression targets: raw teacher {out_dim} dims")

    # ---- model ------------------------------------------------------------ #
    if args.mlm_checkpoint:
        from geo_distill.mlm_student import load_mlm_student

        model, enc_cfg, enc_step = load_mlm_student(
            args.mlm_checkpoint, out_dim, dropout=args.dropout,
            seed=args.seed, expect_vocab=vocab_size)
        if os.path.isfile(args.tokenizer):
            # the vocab-size guard can't tell two different tokenizers of equal
            # vocab apart — with a Hub encoder, prefer its published tokenizer
            print(f"WARNING: --mlm-checkpoint with a *local* tokenizer "
                  f"({args.tokenizer}): make sure it is the exact tokenizer "
                  f"the encoder was pretrained with (for ZurabDz/ka-mlm: "
                  f"--tokenizer ZurabDz/ka-bpe-32k)")
        config = StudentConfig(
            student_type="mlm", vocab_size=vocab_size, out_dim=out_dim,
            max_len=args.max_len, tokenizer=args.tokenizer, center=center,
            mlm_checkpoint=args.mlm_checkpoint, mlm_step=enc_step,
            mlm_encoder=enc_cfg.to_json_dict())
    else:
        model = EmbeddingModel(vocab_size=vocab_size, dim=args.dim, depth=args.depth,
                               heads=args.heads, mlp_dim=args.mlp_dim, max_len=args.max_len,
                               out_dim=out_dim, dropout=args.dropout,
                               embed_dim=args.embed_dim, rngs=nnx.Rngs(args.seed))
        config = StudentConfig(
            student_type="scratch", vocab_size=vocab_size, out_dim=out_dim,
            max_len=args.max_len, tokenizer=args.tokenizer, center=center,
            dim=args.dim, depth=args.depth, heads=args.heads,
            mlp_dim=args.mlp_dim, embed_dim=args.embed_dim)
    # dropout in the MLM encoder follows train()/eval() mode; harmless for the
    # from-scratch student, whose calls pass `deterministic` explicitly
    if args.dropout > 0:
        model.train()
    else:
        model.eval()
    print(f"student parameters: {param_count(model):,}")

    lr = args.lr if args.lr is not None else spec.default_lr
    print(f"learning rate: {lr:g}" + ("" if args.lr is not None else " (default)"))

    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    total_steps = max(2, steps_per_epoch * args.epochs)
    warmup = min(args.warmup, max(1, total_steps // 2))  # never exceed the run
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup, total_steps, end_value=lr * 0.1)
    tx = optax.adamw(schedule, weight_decay=args.weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    deterministic = args.dropout == 0.0
    sim_name = args.sim_loss
    temp = args.temperature
    reg_w, sim_w = args.reg_weight, args.sim_weight

    def compute_loss(model, tok_b, mask_b, teach_b, target_b):
        # When dropout > 0, the model draws from its own stored rng stream;
        # nnx.jit threads that state through, so no key is passed by hand.
        emb = model(tok_b, mask_b, deterministic=deterministic)
        loss = 0.0
        if reg_w > 0:
            loss = loss + reg_w * cosine_regression_loss(emb, target_b)
        if sim_w > 0:
            sim = (mse_sim_loss(emb, teach_b) if sim_name == "mse"
                   else kl_sim_loss(emb, teach_b, temp))
            loss = loss + sim_w * sim
        return loss

    @nnx.jit
    def train_step(model, optimizer, tok_b, mask_b, teach_b, target_b):
        loss, grads = nnx.value_and_grad(compute_loss)(
            model, tok_b, mask_b, teach_b, target_b)
        optimizer.update(model, grads)
        return loss

    # ---- save config (+ the target-space transform) up front so an
    # interrupted run's best params stay usable: eval needs the config (the mlm
    # branch cannot reconstruct the encoder architecture from CLI flags), and a
    # stale config from an earlier run in the same out_dir would otherwise pair
    # with the new params ---------------------------------------------------- #
    os.makedirs(args.out_dir, exist_ok=True)
    save_student_config(args.out_dir, config)
    if center:
        # Without this the space the student was trained into is unreproducible
        # after the run exits: mapping a raw teacher vector into it requires
        # subtracting this exact mean first.
        atomic_np_save(os.path.join(args.out_dir, paths.TEACHER_MEAN),
                       teacher_mean.astype(np.float32))

    # ---- train ------------------------------------------------------------ #
    best_spearman = -1.0
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_idx))
        running = 0.0
        for s in range(steps_per_epoch):
            bidx = order[s * args.batch_size : (s + 1) * args.batch_size]
            loss = train_step(
                model, optimizer,
                jnp.asarray(tr_tokens[bidx]), jnp.asarray(tr_mask[bidx]),
                jnp.asarray(tr_teacher[bidx]), jnp.asarray(tr_target[bidx]))
            running += float(loss)

        # evaluate on the held-out set (dropout off, then back on for training)
        model.eval()
        va_student = embed_in_batches(model, va_tokens, va_mask)
        if args.dropout > 0:
            model.train()
        m = similarity_agreement(va_student, va_teacher)
        print(f"epoch {epoch:3d} | train_loss {running / steps_per_epoch:.5f} "
              f"| val pearson {m['pearson']:.3f} spearman {m['spearman']:.3f} "
              f"nn@1 {m['top1_nn_agreement']:.3f}")

        if m["spearman"] > best_spearman:
            best_spearman = m["spearman"]
            save_student_params(args.out_dir, model)

    # ---- restamp the config with the run's outcome ------------------------- #
    config.best_val_spearman = best_spearman
    save_student_config(args.out_dir, config)
    print(f"Best val Spearman: {best_spearman:.3f}")
    print(f"Saved {paths.STUDENT_PARAMS} + {paths.STUDENT_CONFIG} to {args.out_dir}/")
