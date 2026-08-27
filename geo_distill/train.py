"""Distill the teacher's embedding geometry into the tiny student.

Reads the cached teacher embeddings + the trained tokenizer, then trains the
Flax model with per-example cosine regression: the student outputs directly in
the teacher's full embedding space (out_dim == teacher_dim) and each student
embedding is pulled onto its own teacher vector (LEAF-style, no PCA). The
targets are mean-centered first (raw Gemini vectors share a large common
component; without centering a from-scratch student collapses onto the
centroid). This gives every example a full-information target instead of only
the weak, batch-relative similarity signal.

Optionally (--sim_weight > 0) a within-batch similarity-matching term can be
added on top (KL over soft-retrieval distributions, or MSE on the Gram
matrices), but ablations show pure regression is enough, so it is off by default.

Usage:
    python train.py --out_dir model-v9                     # pure regression, full teacher dim
    python train.py --embed_dim 128 --depth 6 --dropout 0.2 --epochs 40
    python train.py --sim_weight 0.5 --sim_loss kl         # add the aux term back
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx, serialization

from model import EmbeddingModel, param_count
from distill_lib import (
    load_tokenizer, encode_batch, mse_sim_loss, kl_sim_loss, similarity_agreement,
    val_split, cosine_regression_loss,
)


def embed_all(model, tokens, mask, batch=256):
    """Run the student over all rows (deterministic) and stack the embeddings."""
    outs = []
    for i in range(0, tokens.shape[0], batch):
        e = model(jnp.asarray(tokens[i : i + batch]),
                  jnp.asarray(mask[i : i + batch]), deterministic=True)
        outs.append(np.asarray(e))
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    # data / artifacts
    ap.add_argument("--sentences", default="artifacts/sentences.json")
    ap.add_argument("--teacher_emb", default="artifacts/teacher_emb.npy")
    ap.add_argument("--tokenizer", default="artifacts/tokenizer.json",
                    help="tokenizers-JSON path, or a Hub repo id (with "
                         "--mlm_checkpoint you want the tokenizer the encoder "
                         "was pretrained with, e.g. ZurabDz/ka-bpe-32k)")
    ap.add_argument("--out_dir", default="artifacts")
    ap.add_argument("--mlm_checkpoint", default="",
                    help="initialise the student from a pretrained MLM encoder: "
                         "a Hub repo pushed by lm's --hub-checkpoints (e.g. "
                         "ZurabDz/ka-mlm) or a local lm --save-dir. The whole "
                         "encoder is fine-tuned; the from-scratch --dim/--depth/"
                         "--heads/--mlp_dim/--embed_dim flags are ignored")
    # model
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp_dim", type=int, default=512)
    ap.add_argument("--out_dim", type=int, default=None,
                    help="student output width; defaults to the teacher dim so "
                         "the student regresses directly onto raw teacher vectors")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--embed_dim", type=int, default=None,
                    help="token-embedding width; < dim factorizes the table "
                         "(e.g. 128) and frees params for more depth")
    # optim
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 3e-4 from scratch, 5e-5 when fine-tuning a "
                         "pretrained encoder (--mlm_checkpoint)")
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    # loss
    ap.add_argument("--no_center", action="store_true",
                    help="do NOT mean-center the teacher targets. Centering "
                         "(the default) removes the shared component of the "
                         "teacher vectors; without it a from-scratch student "
                         "collapses onto the teacher centroid.")
    ap.add_argument("--reg_weight", type=float, default=1.0,
                    help="weight of the per-example cosine regression onto the "
                         "teacher vector (0 disables it)")
    ap.add_argument("--sim_weight", type=float, default=0.0,
                    help="weight of the within-batch similarity-matching aux "
                         "loss (0 = pure regression, the default)")
    ap.add_argument("--sim_loss", default="kl", choices=["mse", "kl"])
    ap.add_argument("--temperature", type=float, default=0.05)
    args = ap.parse_args()
    assert args.reg_weight > 0 or args.sim_weight > 0, "all loss weights are zero"

    # ---- load data -------------------------------------------------------- #
    with open(args.sentences, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    teacher = np.load(args.teacher_emb).astype(np.float32)
    assert len(sentences) == teacher.shape[0], "sentences / teacher_emb misaligned"
    tok = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    print(f"{len(sentences)} pairs | teacher dim {teacher.shape[1]} | vocab {vocab_size}")

    if args.mlm_checkpoint:
        # the encoder was pretrained on [CLS] ... [SEP] rows; feed it the same
        from mlm_student import encode_batch_mlm

        tokens, mask = encode_batch_mlm(tok, sentences, args.max_len)
    else:
        tokens, mask = encode_batch(tok, sentences, args.max_len)

    # ---- train / val split (stable & content-based so runs stay comparable) #
    train_idx, val_idx = val_split(sentences, args.val_frac, args.seed)
    n_val = len(val_idx)
    rng = np.random.default_rng(args.seed)   # for per-epoch minibatch shuffling

    tr_tokens, tr_mask, tr_teacher = tokens[train_idx], mask[train_idx], teacher[train_idx]
    va_tokens, va_mask, va_teacher = tokens[val_idx], mask[val_idx], teacher[val_idx]

    # ---- regression targets: the teacher vectors, no dim projection -------- #
    # The student outputs in the teacher's full space (out_dim == teacher_dim).
    # We mean-center the targets (train mean only): raw Gemini vectors share a
    # large common component (mean pairwise cosine ~0.67), so without centering
    # a from-scratch student collapses onto that centroid and learns no
    # discriminative geometry. Centering spreads the targets over the sphere.
    teacher_dim = teacher.shape[1]
    out_dim = teacher_dim if args.out_dim is None else args.out_dim
    if out_dim != teacher_dim:
        raise SystemExit(
            f"--out_dim {out_dim} != teacher dim {teacher_dim}; cosine regression "
            f"needs matching dims. Omit --out_dim to match the teacher.")
    if args.no_center:
        tr_target = tr_teacher
        print(f"regression targets: raw teacher {teacher_dim} dims (out_dim {out_dim})")
    else:
        teacher_mean = tr_teacher.mean(axis=0, keepdims=True)
        centered = tr_teacher - teacher_mean
        tr_target = centered / (np.linalg.norm(centered, axis=-1, keepdims=True) + 1e-8)
        print(f"regression targets: mean-centered teacher {teacher_dim} dims "
              f"(out_dim {out_dim})")

    # ---- model ------------------------------------------------------------ #
    if args.mlm_checkpoint:
        from mlm_student import load_mlm_student, encoder_config_to_dict

        model, enc_cfg, enc_step = load_mlm_student(
            args.mlm_checkpoint, out_dim, dropout=args.dropout,
            seed=args.seed, expect_vocab=vocab_size)
        if os.path.isfile(args.tokenizer):
            # the size guard can't tell two different tokenizers of equal
            # vocab apart — with a Hub encoder, prefer its published tokenizer
            print(f"WARNING: --mlm_checkpoint with a *local* tokenizer "
                  f"({args.tokenizer}): make sure it is the exact tokenizer "
                  f"the encoder was pretrained with (for ZurabDz/ka-mlm: "
                  f"--tokenizer ZurabDz/ka-bpe-32k)")
    else:
        model = EmbeddingModel(vocab_size=vocab_size, dim=args.dim, depth=args.depth,
                               heads=args.heads, mlp_dim=args.mlp_dim, max_len=args.max_len,
                               out_dim=out_dim, dropout=args.dropout,
                               embed_dim=args.embed_dim, rngs=nnx.Rngs(args.seed))
    # dropout in the MLM encoder follows train()/eval() mode; harmless for the
    # from-scratch student, whose calls pass `deterministic` explicitly
    model.train() if args.dropout > 0 else model.eval()
    print(f"student parameters: {param_count(model):,}")

    lr = args.lr if args.lr is not None else (5e-5 if args.mlm_checkpoint else 3e-4)
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

    # ---- save config up front so an interrupted run's best params stay
    # usable: eval.py needs it (the mlm branch cannot reconstruct the encoder
    # architecture from CLI flags), and a stale config from an earlier run in
    # the same out_dir would otherwise pair with the new params ------------- #
    os.makedirs(args.out_dir, exist_ok=True)
    config = dict(vocab_size=vocab_size, dim=args.dim, depth=args.depth,
                  heads=args.heads, mlp_dim=args.mlp_dim, out_dim=out_dim,
                  max_len=args.max_len, dropout=0.0, embed_dim=args.embed_dim,
                  tokenizer=args.tokenizer)
    if args.mlm_checkpoint:
        config.update(student_type="mlm", mlm_checkpoint=args.mlm_checkpoint,
                      mlm_step=enc_step, mlm_encoder=encoder_config_to_dict(enc_cfg))
    with open(os.path.join(args.out_dir, "student_config.json"), "w") as f:
        json.dump(config, f, indent=2)

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
        va_student = embed_all(model, va_tokens, va_mask)
        if args.dropout > 0:
            model.train()
        m = similarity_agreement(va_student, va_teacher)
        print(f"epoch {epoch:3d} | train_loss {running / steps_per_epoch:.5f} "
              f"| val pearson {m['pearson']:.3f} spearman {m['spearman']:.3f} "
              f"nn@1 {m['top1_nn_agreement']:.3f}")

        if m["spearman"] > best_spearman:
            best_spearman = m["spearman"]
            # atomic: a kill mid-write must not tear the previous best
            path = os.path.join(args.out_dir, "student_params.msgpack")
            with open(path + ".tmp", "wb") as f:
                f.write(serialization.to_bytes(nnx.to_pure_dict(nnx.state(model, nnx.Param))))
            os.replace(path + ".tmp", path)

    # ---- restamp the config with the run's outcome ------------------------- #
    config["best_val_spearman"] = best_spearman
    with open(os.path.join(args.out_dir, "student_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Best val Spearman: {best_spearman:.3f}")
    print(f"Saved student_params.msgpack + student_config.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
