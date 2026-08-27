"""The jitted steps and the training loop."""

import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mlm import checkpoint as ckpt
from mlm import hub
from mlm import progress
from mlm.config import N_SPECIAL, PAD_ID, EncoderConfig
from mlm.data import decode_record, load_sources, make_dataset
from mlm.masking import n_predictions
from mlm.model import MlmEncoder, mlm_loss, count_params
from mlm.optim import make_optimizer, trapezoid_schedule
from mlm.progress import bar, restart_timer
from mlm.sharding import device_stream, make_shardings, replicate, to_device


# nnx.jit adopts each argument's own sharding when the array is committed, so no
# in_shardings is needed here — device_put in to_device() does the committing.
# Donation lets XLA alias the input param/moment buffers into the outputs; it is
# a no-op on CPU and a real memory win on TPU/GPU.
@nnx.jit(donate_argnames=("model", "optimizer"))
def train_step(model: MlmEncoder, optimizer: nnx.Optimizer, batch):
    def loss_fn(m):
        return mlm_loss(m, batch)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: MlmEncoder, batch):
    return mlm_loss(model, batch)


def evaluate(model, val_ds, max_batches: int = 32):
    """Mean MLM loss over held-out windows.

    No re-seeding needed any more: masks are a pure function of element index,
    so two evals over the same finite dataset are comparable by construction.
    Capped at max_batches so eval cost stays flat as the split grows.
    """
    losses = []
    for batch in val_ds:
        if len(losses) >= max_batches:
            break
        losses.append(eval_step(model, batch))
    if not losses:
        return 0.0
    return float(sum(losses) / len(losses))  # one sync, not one per batch


def train(args) -> None:
    train_source, val_source, decode, vocab_size = load_sources(args)

    # Built before the model because resuming has to rebuild the optimizer with
    # the *same* tx, and the schedule is baked into it.
    schedule = trapezoid_schedule(args.peak_lr, args.steps)
    data_sharding, repl = make_shardings(args.batch_size)

    if args.resume:
        if not args.save_dir:
            raise RuntimeError("--resume needs a --save-dir to resume from")
        if args.hub_checkpoints and not args.smoke:
            # A fresh session pulls the repo's checkpoint; a local dir that is
            # already ahead (or tied) is left alone. (cli already vetoes the
            # --smoke combination; the guard here is defense in depth.)
            hub.pull_checkpoint(args.save_dir, args.hub_checkpoints)
        model, optimizer, cfg, start_step = ckpt.resume_checkpoint(args.save_dir, schedule)
        if cfg.vocab_size != vocab_size:
            raise RuntimeError(
                f"checkpoint has vocab {cfg.vocab_size} but the tokenizer has {vocab_size}"
            )
        if start_step >= args.steps:
            raise RuntimeError(
                f"checkpoint is at step {start_step} and --steps is {args.steps}; "
                f"--steps is the *total* run length and defines the LR schedule, "
                f"so raise it to continue"
            )
        print(f"resumed from step {start_step} in {os.path.abspath(args.save_dir)}")
    else:
        if args.save_dir:
            ckpt.assert_writable(args.save_dir)
            if args.hub_checkpoints and not args.smoke:
                # the Hub-side twin of assert_writable: a fresh run would push
                # step 500 over a repo already holding a later checkpoint
                hub_step = hub.hub_latest_step(args.hub_checkpoints)
                if hub_step is not None:
                    raise RuntimeError(
                        f"hf.co/{hub.strip_prefix(args.hub_checkpoints)} already "
                        f"holds a checkpoint at step {hub_step}. Pass --resume to "
                        f"continue it (it is pulled automatically), or push to a "
                        f"different repo."
                    )
        cfg = EncoderConfig(
            vocab_size=vocab_size,
            hidden=args.hidden,
            layers=args.layers,
            heads=args.heads,
            mlp_hidden=int(round(args.hidden * 8 / 3 / 64)) * 64,
            max_len=args.seq_len,
            dropout=args.dropout,
            dtype=getattr(jnp, args.dtype),
            remat=args.remat,
        )
        model = MlmEncoder(cfg, rngs=nnx.Rngs(args.seed))
        model.train()
        optimizer = make_optimizer(model, schedule)
        start_step = 0

    replicate(model, optimizer, repl)
    if repl is not None:
        print(f"data-parallel over {jax.device_count()} devices, "
              f"{args.batch_size // jax.device_count()} sequences each")

    counts = count_params(model)
    print(f"params: {counts['total'] / 1e6:.2f}M total, "
          f"{counts['non_embedding'] / 1e6:.2f}M non-embedding, "
          f"{counts['embedding'] / 1e6:.2f}M in the tied table "
          f"({cfg.dtype.__name__} compute, {cfg.param_dtype.__name__} params)")

    mgr = ckpt.checkpoint_manager(args.save_dir, args.keep) if args.save_dir else None
    if mgr is not None and not args.resume:
        # Only on a fresh run: rewriting this on resume would clobber the config
        # that the checkpoints already in the directory were trained under.
        ckpt.save_config(args.save_dir, cfg)

    n_pred = n_predictions(args.seq_len, args.mask_prob)
    train_ds = make_dataset(train_source, args.batch_size, args.mask_prob, n_pred,
                            args.seed + 1, decode=decode)
    val_ds = make_dataset(val_source, args.batch_size, args.mask_prob, n_pred,
                          args.seed + 2, decode=decode, repeat=False)
    data_iter = iter(device_stream(train_ds, data_sharding))
    if args.resume:
        # The whole point of checkpointing the iterator: restore the exact
        # position rather than reseeding into a different stream.
        ckpt.restore_data_iter(args.save_dir, data_iter, start_step)

    tokens_per_step = args.batch_size * args.seq_len
    print(f"training steps {start_step + 1}..{args.steps} at {args.mask_prob:.0%} masking, "
          f"{tokens_per_step} tokens/step")
    # The first step jit-compiles the whole graph — minutes of silence on a GPU
    # if nothing says so. Progress lines start once step 1 is actually done.
    print("compiling the train step — the first step takes a minute or two ...",
          flush=True)

    start = time.time()
    timed_from = start_step
    last_pushed = None
    pb = bar(total=args.steps, desc="train", unit="step", initial=start_step)
    for step in range(start_step + 1, args.steps + 1):
        batch = next(data_iter)
        loss = train_step(model, optimizer, batch)
        pb.update(1)
        if step == start_step + 1:
            # Step 1 is where jit compiles the whole graph. Block on it, report
            # it, then restart every clock: folded into the run average, a
            # 90-second compile would understate tok/s by 100x at step 1 and
            # still ~40% at step 500, with the ETA wrong the same way.
            pb.write(f"step {step:>6}  loss {float(loss):6.3f}  "
                     f"(first step took {time.time() - start:.0f}s, jit compile included)")
            start = time.time()
            timed_from = step
            restart_timer(pb)
        elif step % args.log_every == 0:
            # float(loss) is a device sync, so it stays on the log cadence
            # rather than happening every step just to feed the bar.
            loss = float(loss)
            elapsed = time.time() - start
            done = step - timed_from  # steps inside the timed window
            rate = done * tokens_per_step / max(elapsed, 1e-6) / 1e3
            lr = float(schedule(step - 1))
            if progress.PROGRESS:
                pb.set_postfix(loss=f"{loss:.3f}",
                               ppl=f"{math.exp(min(loss, 20)):.1f}",
                               lr=f"{lr:.2e}", tok_s=f"{rate:.1f}k")
            else:
                print(f"step {step:>6}  loss {loss:6.3f}  "
                      f"ppl {math.exp(min(loss, 20)):9.1f}  "
                      f"lr {lr:.2e}  {rate:6.1f}k tok/s", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            # `deterministic` is static, so toggling it compiles a second copy of
            # the whole 12-layer graph. At dropout 0.0 the two are identical, so
            # skip the toggle and the extra compile with it.
            if cfg.dropout > 0.0:
                model.eval()
            val = evaluate(model, device_stream(val_ds, data_sharding))
            if cfg.dropout > 0.0:
                model.train()
            pb.write(f"step {step:>6}  held-out loss {val:6.3f}  (train {float(loss):6.3f})")
        if mgr is not None and args.save_every and step % args.save_every == 0:
            ckpt.save_checkpoint(mgr, step, model, optimizer, data_iter)
            if (args.hub_checkpoints and not args.smoke and args.push_every
                    and step % args.push_every == 0):
                # orbax saves asynchronously; only complete files may upload
                mgr.wait_until_finished()
                try:
                    hub.push_checkpoint(args.save_dir, step, args.hub_checkpoints)
                    last_pushed = step
                except Exception as e:
                    # a Hub hiccup must not kill an hours-long run; the local
                    # checkpoint exists and the next push retries naturally
                    pb.write(f"WARNING: checkpoint push failed at step {step}: {e}")

    pb.close()
    model.eval()
    final = evaluate(model, device_stream(val_ds, data_sharding))
    print(f"final held-out mlm loss {final:.3f}")

    if mgr is not None:
        ckpt.save_checkpoint(mgr, args.steps, model, optimizer, data_iter)
        mgr.wait_until_finished()
        print(f"checkpoints in {os.path.abspath(args.save_dir)}: steps {mgr.all_steps()}")
        mgr.close()
        if args.hub_checkpoints and not args.smoke and last_pushed != args.steps:
            # the final push raises on failure: training is already saved
            # locally, and a silent miss here would read as "it's on the Hub"
            hub.push_checkpoint(args.save_dir, args.steps, args.hub_checkpoints)

    # embeddings for downstream use: mean-pool the hidden states over content
    # tokens only. `>= N_SPECIAL` drops [PAD], [CLS], [SEP] and [MASK] at once —
    # [CLS] in particular is an untrained attention sink here, so averaging it in
    # would fold a high-norm outlier into every embedding. Fed clean windows,
    # not masked ones: corrupting the input is a training device, not a step you
    # want between raw text and its vector.
    #
    # This is a smoke check that the encode path works, not evidence that the
    # vectors are good. MLM puts no loss on the aggregate of a sequence, so the
    # pooled space comes out anisotropic and similarity is dominated by token
    # frequency — a contrastive stage is what actually shapes it.
    rows = np.stack([np.asarray(val_source[i]) if not decode
                     else decode_record(val_source[i])
                     for i in range(args.batch_size)])
    batch = to_device({"ids": rows, "am": (rows != PAD_ID).astype(np.int32)}, data_sharding)
    hidden = nnx.jit(lambda m, i, a: m.encode(i, a))(model, batch["ids"], batch["am"])
    w = (batch["ids"] >= N_SPECIAL)[..., None].astype(hidden.dtype)
    pooled = (hidden * w).sum(1) / jnp.maximum(w.sum(1), 1.0)
    print(f"pooled embedding shape {pooled.shape}")
