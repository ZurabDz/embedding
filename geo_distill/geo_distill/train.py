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

The loop is resumable, because ten epochs over a million sentences outlive a
free Kaggle session. Every --save-every epochs it writes a checkpoint next to
the best parameters — the *latest* parameters, the Adam moments (which carry
the learning-rate schedule's position), the dropout rng counters and the
shuffle position — and --resume picks it up. With --push-to, that checkpoint is
mirrored to a private Hub model repo and pulled back when a fresh session finds
it ahead of local, so the same command chains across sessions the way lm's
--hub-checkpoints does for pretraining.
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import jax.numpy as jnp
import optax
from flax import nnx

from geo_distill import config as paths
from geo_distill import hub
from geo_distill.checkpoint import (atomic_np_save, load_student_state,
                                    load_train_state, save_student_config,
                                    save_student_params, save_student_state,
                                    save_train_state)
from geo_distill.data import load_tokenizer, val_split
from geo_distill.losses import cosine_regression_loss, kl_sim_loss, mse_sim_loss
from geo_distill.metrics import similarity_agreement
from geo_distill.model import EmbeddingModel, embed_in_batches, param_count
from geo_distill.students import StudentConfig, spec_for


# --------------------------------------------------------------------------- #
# Resume bookkeeping
# --------------------------------------------------------------------------- #
def _sha256_file(path: str) -> str:
    """Content identity of an input file.

    Only sentences.json is hashed, never teacher_emb.npy: the two are written
    together by the teacher stage and aligned row-for-row, so any change to the
    corpus shows up here, while hashing a multi-gigabyte matrix would cost a
    full extra read on every startup. A re-embedding of the *same* sentences by
    a different teacher slips through this and is caught by the teacher_mean
    comparison instead.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hms(seconds: float) -> str:
    """h:mm:ss — an ETA is read at a glance, not parsed."""
    s = max(0, int(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _fingerprint(args, *, lr: float, vocab_size: int, n_sentences: int,
                 n_train: int, teacher_dim: int, sentences_sha: str) -> dict:
    """Everything whose change would make a --resume a different run.

    Two kinds live here. Some fields (the architecture, the tokenizer, the
    teacher width) would break the restore outright — but only much later and
    with an unrecognisable error, because msgpack carries no shapes of its own
    and takes them from the blob. The rest (the objective weights, the seed, the
    batch size, the corpus size) restore perfectly and then quietly train
    something that is not the run you asked to continue: they feed the LR
    schedule, the shuffle stream, the target space or the loss.

    --epochs is deliberately absent. Raising it is the one legitimate way to
    change a resumed run, exactly as --steps is for lm.
    """
    scratch = not args.mlm_checkpoint
    return {
        "student_type": "scratch" if scratch else "mlm",
        "mlm_checkpoint": args.mlm_checkpoint,
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "max_len": args.max_len,
        # The from-scratch sizing flags are ignored with --mlm-checkpoint, so
        # they must not be able to fail a resume that never read them.
        "dim": args.dim if scratch else None,
        "depth": args.depth if scratch else None,
        "heads": args.heads if scratch else None,
        "mlp_dim": args.mlp_dim if scratch else None,
        "embed_dim": args.embed_dim if scratch else None,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": lr,
        "weight_decay": args.weight_decay,
        "warmup": args.warmup,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "center": not args.no_center,
        "reg_weight": args.reg_weight,
        "sim_weight": args.sim_weight,
        "sim_loss": args.sim_loss,
        "temperature": args.temperature,
        "n_sentences": n_sentences,
        "n_train": n_train,
        "teacher_dim": teacher_dim,
        "sentences_sha256": sentences_sha,
    }


# Fields whose mismatch has a cause worth naming, because the flag did not move
# — the data underneath it did.
_DATA_FIELDS = ("n_sentences", "n_train", "sentences_sha256", "teacher_dim")


def _check_fingerprint(saved: dict, current: dict) -> None:
    diffs = [k for k in current if saved.get(k) != current[k]]
    if not diffs:
        return
    lines = [f"    {k}: {saved.get(k)!r} -> {current[k]!r}" for k in diffs]
    why = ("Restore the original values, or drop --resume to start a fresh run "
           "(--epochs is the one flag you may raise on a resume).")
    if any(k in _DATA_FIELDS for k in diffs):
        why = ("The corpus itself changed. A longer corpus rescales the whole "
               "learning-rate schedule, redefines what an epoch is, and moves "
               "the mean the regression targets are centered on — so the saved "
               "parameters would be chasing a target space that shifted under "
               "them. Continue on the original data, or start a fresh run on "
               "the new data.")
    raise RuntimeError(
        "--resume: these settings differ from the run being resumed:\n"
        + "\n".join(lines) + "\n\n"
        + "They define the training math, so continuing would silently produce "
          "a different run. " + why)


def _resume_prelude(args) -> dict | None:
    """The Hub sync and the checks that must happen before any heavy work.

    Returns the train state to continue from, or None for a fresh run. Both
    branches touch the network at most once and both fail here rather than
    minutes in: loading a million sentences and pulling an encoder only to
    discover the repo already holds a finished run is a bad trade.
    """
    if not args.resume:
        if args.push_to:
            # The Hub-side twin of "don't clobber": a fresh run would replace
            # the repo's checkpoint wholesale, and unlike the local directory
            # there is no copy of it anywhere else.
            remote = hub.hub_train_state(args.push_to)
            if remote is not None:
                raise RuntimeError(
                    f"hf.co/{hub.strip_prefix(args.push_to)} already holds a "
                    f"run at epoch {remote.get('epoch')}/{remote.get('epochs')} "
                    f"(best val spearman {remote.get('best_val_spearman')}). "
                    f"Pass --resume to continue it — it is pulled "
                    f"automatically — or push to a different repo.")
        return None

    if args.push_to:
        hub.pull_student_for_resume(args.out_dir, args.push_to)
    prior = load_train_state(args.out_dir)
    if prior is None:
        where = os.path.abspath(args.out_dir)
        if args.push_to:
            where += f" or on hf.co/{hub.strip_prefix(args.push_to)}"
        raise RuntimeError(
            f"--resume found no {paths.TRAIN_STATE} in {where} — this looks "
            f"like the first session, so run the same command without --resume.")
    for name in (paths.STUDENT_CONFIG, paths.STUDENT_STATE):
        if not os.path.isfile(os.path.join(args.out_dir, name)):
            raise RuntimeError(
                f"--resume: {os.path.abspath(args.out_dir)} has "
                f"{paths.TRAIN_STATE} but no {name}, so the checkpoint is "
                f"incomplete. Start over without --resume.")
    return prior


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def run(args) -> None:
    import json

    os.makedirs(args.out_dir, exist_ok=True)
    prior = _resume_prelude(args)

    # ---- load data -------------------------------------------------------- #
    with open(args.sentences, "r", encoding="utf-8") as f:
        sentences = json.load(f)
    teacher = np.load(args.teacher_emb).astype(np.float32)
    assert len(sentences) == teacher.shape[0], "sentences / teacher_emb misaligned"
    tok = load_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    print(f"{len(sentences)} pairs | teacher dim {teacher.shape[1]} | vocab {vocab_size}")

    spec = spec_for("mlm" if args.mlm_checkpoint else "scratch")

    # ---- train / val split (stable & content-based so runs stay comparable) #
    train_idx, val_idx = val_split(sentences, args.val_frac, args.seed)
    rng = np.random.default_rng(args.seed)   # for per-epoch minibatch shuffling

    # The fingerprint is checked before the expensive tokenization, so a
    # mismatched resume costs seconds rather than minutes.
    lr = args.lr if args.lr is not None else spec.default_lr
    fingerprint = _fingerprint(
        args, lr=lr, vocab_size=vocab_size, n_sentences=len(sentences),
        n_train=len(train_idx), teacher_dim=teacher.shape[1],
        sentences_sha=_sha256_file(args.sentences))
    if prior is not None:
        _check_fingerprint(prior.get("fingerprint", {}), fingerprint)

    # Hard contract check, not a warning: a tokenizer whose special ids don't
    # match the student's row layout silently corrupts every row.
    spec.validate_tokenizer(tok)
    tokens, mask = spec.encode(tok, sentences, args.max_len)

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
    teacher_mean = None
    if center:
        teacher_mean = tr_teacher.mean(axis=0, keepdims=True)
        centered = tr_teacher - teacher_mean
        tr_target = centered / (np.linalg.norm(centered, axis=-1, keepdims=True) + 1e-8)
        print(f"regression targets: mean-centered teacher {out_dim} dims")
    else:
        tr_target = tr_teacher
        print(f"regression targets: raw teacher {out_dim} dims")

    mean_path = os.path.join(args.out_dir, paths.TEACHER_MEAN)
    if prior is not None and center and os.path.isfile(mean_path):
        # The one content check on the teacher matrix itself (the fingerprint
        # only hashes the sentences). If this moved, every saved parameter was
        # trained toward a different target space.
        saved_mean = np.load(mean_path)
        if saved_mean.shape != teacher_mean.shape or not np.allclose(
                saved_mean, teacher_mean, rtol=1e-4, atol=1e-6):
            raise RuntimeError(
                f"--resume: the teacher embeddings changed — the training-split "
                f"mean in {mean_path} does not match the one recomputed from "
                f"{args.teacher_emb}. The saved parameters were trained into "
                f"the old space, so continuing would train them toward a "
                f"different one. Start a fresh run on the new embeddings.")

    # ---- model ------------------------------------------------------------ #
    if prior is not None:
        # Rebuilt from the saved config rather than from --mlm-checkpoint: the
        # architecture is fully described on disk, so a resumed session never
        # re-downloads the pretrained encoder just to overwrite its weights.
        config = StudentConfig.load(os.path.join(args.out_dir, paths.STUDENT_CONFIG))
        model = spec.rebuild(config, dropout=args.dropout, seed=args.seed)
        print(f"resuming the {config.student_type} student from "
              f"{os.path.abspath(args.out_dir)}")
    elif args.mlm_checkpoint:
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
            dropout=args.dropout, mlm_checkpoint=args.mlm_checkpoint,
            mlm_step=enc_step, mlm_encoder=enc_cfg.to_json_dict())
    else:
        model = EmbeddingModel(vocab_size=vocab_size, dim=args.dim, depth=args.depth,
                               heads=args.heads, mlp_dim=args.mlp_dim, max_len=args.max_len,
                               out_dim=out_dim, dropout=args.dropout,
                               embed_dim=args.embed_dim, rngs=nnx.Rngs(args.seed))
        config = StudentConfig(
            student_type="scratch", vocab_size=vocab_size, out_dim=out_dim,
            max_len=args.max_len, tokenizer=args.tokenizer, center=center,
            dropout=args.dropout, dim=args.dim, depth=args.depth,
            heads=args.heads, mlp_dim=args.mlp_dim, embed_dim=args.embed_dim)
    # Dropout in the MLM encoder follows train()/eval() mode (its `deterministic`
    # argument is inert), and rebuild() hands the module back in eval mode — so
    # this block is load-bearing for a resumed mlm run, not just cosmetic.
    if args.dropout > 0:
        model.train()
    else:
        model.eval()
    print(f"student parameters: {param_count(model):,}")
    print(f"learning rate: {lr:g}" + ("" if args.lr is not None else " (default)"))

    # ---- schedule --------------------------------------------------------- #
    steps_per_epoch = max(1, len(train_idx) // args.batch_size)
    total_steps = max(2, steps_per_epoch * args.epochs)
    warmup = min(args.warmup, max(1, total_steps // 2))  # never exceed the run
    start_epoch, best_spearman = 0, -1.0
    if prior is not None:
        start_epoch = int(prior["epoch"])
        best_spearman = float(prior.get("best_val_spearman", -1.0))
        if start_epoch >= args.epochs:
            raise RuntimeError(
                f"the run in {os.path.abspath(args.out_dir)} has already "
                f"finished {start_epoch} epochs and --epochs is {args.epochs}. "
                f"--epochs is the *total* run length and defines the LR "
                f"schedule, so raise it to continue.")
        if args.epochs == prior.get("epochs"):
            # Keep the exact schedule the run started on rather than one
            # recomputed from numbers that only happen to agree.
            total_steps, warmup = int(prior["total_steps"]), int(prior["warmup"])
        else:
            print(f"--epochs {args.epochs} (was {prior.get('epochs')}): the "
                  f"cosine decay is re-stretched over {total_steps} steps, so "
                  f"the learning rate steps back up at resume")
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, lr, warmup, total_steps, end_value=lr * 0.1)
    tx = optax.adamw(schedule, weight_decay=args.weight_decay)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    if prior is not None:
        # Parameters, Adam moments and the dropout counters in one blob; the
        # optimizer's own step counts are what put the schedule back where it
        # was, so this is also what stops a resume from re-warming up.
        load_student_state(args.out_dir, model, optimizer)
        # ...and the shuffle stream, so the resumed epochs see the permutations
        # the uninterrupted run would have seen. Restored verbatim: the
        # generator refuses a hand-built state dict.
        rng.bit_generator.state = prior["numpy_rng"]
        print(f"resumed at epoch {start_epoch}/{args.epochs} "
              f"(best val spearman {best_spearman:.3f}, "
              f"lr {float(schedule(steps_per_epoch * start_epoch)):.2e})")

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
    config.best_val_spearman = best_spearman if prior is not None else None
    save_student_config(args.out_dir, config)
    if center:
        # Without this the space the student was trained into is unreproducible
        # after the run exits: mapping a raw teacher vector into it requires
        # subtracting this exact mean first.
        atomic_np_save(mean_path, teacher_mean.astype(np.float32))
    if prior is None:
        # A fresh run replaces whatever was here, so drop any previous run's
        # resume state now: a directory holding one run's checkpoint next to
        # another run's parameters is the one state --resume cannot detect.
        for name in (paths.STUDENT_STATE, paths.TRAIN_STATE):
            stale = os.path.join(args.out_dir, name)
            if os.path.isfile(stale):
                os.remove(stale)

    def checkpoint(epoch_done: int) -> None:
        """One consistent snapshot: best params (already written), latest
        params + moments + rng, and where the run is."""
        config.best_val_spearman = best_spearman
        save_student_config(args.out_dir, config)
        save_student_state(args.out_dir, model, optimizer)
        save_train_state(args.out_dir, {
            "epoch": epoch_done,
            "epochs": args.epochs,
            "best_val_spearman": best_spearman,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup": warmup,
            "lr": lr,
            # Verbatim, all four keys: the generator rejects a hand-built dict,
            # and dropping the cached-half-word entries shifts the stream.
            "numpy_rng": rng.bit_generator.state,
            "fingerprint": fingerprint,
        })

    # ---- train ------------------------------------------------------------ #
    print(f"training epochs {start_epoch + 1}..{args.epochs} at "
          f"{steps_per_epoch} steps/epoch ({total_steps} total)")
    if args.log_every:
        # On the real corpus an epoch is thousands of steps, so without this
        # the run looks hung between the (minutes or hours apart) epoch lines.
        print("compiling the train step — the first step takes a moment ...",
              flush=True)
    pushed = None
    # The in-epoch log reports the window since the previous line, not the run
    # so far: a rate averaged from step 1 would still be carrying the compile,
    # and a loss averaged from step 1 stops moving once the epoch is old.
    seen = start_epoch * steps_per_epoch     # global steps, resume included
    for epoch in range(start_epoch, args.epochs):
        order = rng.permutation(len(train_idx))
        running = 0.0
        # Opened per epoch, so the validation pass and any Hub push between two
        # epochs never land inside a window's clock.
        win_t0, win_seen, win_loss = time.time(), seen, 0.0
        for s in range(steps_per_epoch):
            bidx = order[s * args.batch_size : (s + 1) * args.batch_size]
            # float() is a device sync; the epoch average needs it every step
            # anyway, so the log line costs nothing extra.
            loss = float(train_step(
                model, optimizer,
                jnp.asarray(tr_tokens[bidx]), jnp.asarray(tr_mask[bidx]),
                jnp.asarray(tr_teacher[bidx]), jnp.asarray(tr_target[bidx])))
            running += loss
            win_loss += loss
            seen += 1
            if not args.log_every:
                continue
            if epoch == start_epoch and s == 0:
                # jit compiled the whole graph inside this one step. Report it
                # and restart the clock: folded into the average it would
                # understate the rate for the rest of the session and skew
                # every ETA with it.
                print(f"  first step took {time.time() - win_t0:.0f}s "
                      f"(jit compile included)", flush=True)
                win_t0, win_seen, win_loss = time.time(), seen, 0.0
            elif (s + 1) % args.log_every == 0:
                now = time.time()
                n = seen - win_seen
                rate = n / max(now - win_t0, 1e-6)
                # Training steps only: the per-epoch validation, checkpoint and
                # push are not in the rate, so this reads a little optimistic.
                left = (args.epochs - epoch - 1) * steps_per_epoch + \
                    steps_per_epoch - (s + 1)
                print(f"  epoch {epoch:3d} step {s + 1:>6}/{steps_per_epoch} "
                      f"| loss {win_loss / n:.5f} "
                      f"| lr {float(schedule(seen - 1)):.2e} "
                      f"| {rate:.2f} it/s | eta {_hms(left / max(rate, 1e-9))}",
                      flush=True)
                win_t0, win_seen, win_loss = now, seen, 0.0

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

        done = epoch + 1
        if done == args.epochs or (args.save_every and done % args.save_every == 0):
            checkpoint(done)
        if args.push_to and args.push_every and done % args.push_every == 0:
            try:
                hub.push_student(args.out_dir, args.push_to, epoch=done,
                                 epochs=args.epochs, best=best_spearman)
                pushed = done
            except Exception as e:  # noqa: BLE001
                # A Hub hiccup must not kill a long run: the local checkpoint
                # exists and the next push retries naturally.
                print(f"WARNING: student push failed at epoch {done}: {e}")

    # ---- restamp the config with the run's outcome ------------------------- #
    config.best_val_spearman = best_spearman
    save_student_config(args.out_dir, config)
    print(f"Best val Spearman: {best_spearman:.3f}")
    print(f"Saved {paths.STUDENT_PARAMS} (best) + {paths.STUDENT_CONFIG} + "
          f"{paths.STUDENT_STATE} + {paths.TRAIN_STATE} to {args.out_dir}/")
    if args.push_to and pushed != args.epochs:
        # The final push raises on failure: training is already saved locally,
        # and a silent miss here would read as "it is on the Hub".
        hub.push_student(args.out_dir, args.push_to, epoch=args.epochs,
                         epochs=args.epochs, best=best_spearman)
