"""Orbax checkpointing: params, Adam moments, and the data stream's position."""

import dataclasses
import json
import os

import grain
import jax.numpy as jnp
from flax import nnx

from mlm.config import EncoderConfig
from mlm.model import MlmEncoder
from mlm.optim import make_optimizer


def save_config(save_dir: str, cfg: EncoderConfig) -> None:
    """The config is not part of the orbax tree, and without it there is nothing
    to rebuild the model into before restoring."""
    d = dataclasses.asdict(cfg)
    for k in ("dtype", "param_dtype"):
        d[k] = getattr(cfg, k).__name__
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(d, f, indent=2)


def load_config(save_dir: str) -> EncoderConfig:
    with open(os.path.join(save_dir, "config.json")) as f:
        d = json.load(f)
    # param_dtype is absent from configs written before it was split out; the
    # dataclass default (fp32) is the right answer for those.
    for k in ("dtype", "param_dtype"):
        if k in d:
            d[k] = getattr(jnp, d[k])
    return EncoderConfig(**d)


def checkpoint_manager(save_dir: str, keep: int = 10):
    import orbax.checkpoint as ocp

    os.makedirs(save_dir, exist_ok=True)
    return ocp.CheckpointManager(
        os.path.abspath(save_dir),  # orbax refuses relative paths
        options=ocp.CheckpointManagerOptions(max_to_keep=keep, create=True),
    )


def assert_writable(save_dir: str) -> None:
    """Refuse to train into a directory that already holds checkpoints.

    A fresh CheckpointManager over a populated directory silently *drops* every
    save below the highest step already present — it returns False and writes
    nothing. A run started this way looks healthy for hours and then has no
    checkpoints to show for it, so fail here instead.
    """
    if not os.path.isdir(save_dir):
        return
    mgr = checkpoint_manager(save_dir)
    steps = mgr.all_steps()
    mgr.close()
    if steps:
        raise RuntimeError(
            f"{os.path.abspath(save_dir)} already holds checkpoints at steps {sorted(steps)}. "
            f"orbax would silently discard every save below step {max(steps)}. "
            f"Pass --resume to continue that run, or point --save-dir somewhere empty."
        )


def save_checkpoint(mgr, step: int, model: MlmEncoder, optimizer,
                    data_iter=None) -> None:
    """Params, optimizer state, *and* the input pipeline's position.

    The trapezoid plateau is only a free parameter if a fork can resume Adam's
    moments, and a resumed run is only equivalent to an uninterrupted one if the
    data stream picks up where it stopped rather than reseeding into a different
    permutation.
    """
    import orbax.checkpoint as ocp

    items = {
        "model": ocp.args.StandardSave(nnx.state(model, nnx.Param)),
        "opt": ocp.args.StandardSave(nnx.state(optimizer)),
    }
    if data_iter is not None:
        items["data_iter"] = grain.checkpoint.CheckpointSave(item=data_iter)
    mgr.save(step, args=ocp.args.Composite(**items))


def restore_data_iter(save_dir: str, data_iter, step: int) -> None:
    """Restore the grain iterator in place, if this checkpoint carries one.

    Checkpoints written before the pipeline moved to grain have no `data_iter`
    entry; those still resume their weights, they just restart the stream.
    """
    import orbax.checkpoint as ocp

    mgr = checkpoint_manager(save_dir)
    try:
        mgr.restore(step, args=ocp.args.Composite(
            data_iter=grain.checkpoint.CheckpointRestore(item=data_iter)))
        print(f"  data pipeline resumed at its saved position")
    except (KeyError, FileNotFoundError, ValueError) as e:
        print(f"  no data_iter in checkpoint {step} ({type(e).__name__}); "
              f"restarting the stream")
    finally:
        mgr.close()


def load_checkpoint(save_dir: str, step: int | None = None):
    """Rebuild an inference-ready model from disk. Returns (model, cfg, step)."""
    import orbax.checkpoint as ocp

    cfg = load_config(save_dir)
    model = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    mgr = checkpoint_manager(save_dir)
    step = mgr.latest_step() if step is None else step
    if step is None:
        raise RuntimeError(f"no checkpoints found in {save_dir}")
    restored = mgr.restore(step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(nnx.state(model, nnx.Param)),
    ))
    nnx.update(model, restored["model"])
    mgr.close()
    model.eval()
    return model, cfg, step


def resume_checkpoint(save_dir: str, schedule, step: int | None = None):
    """Rebuild model *and* optimizer, Adam moments included.

    StandardRestore restores into a target tree, so the optimizer has to be
    constructed first — with the same `tx`, which means the same schedule built
    from the same total step count. Resuming onto a different schedule would
    silently change the run, so the caller checks that before calling this.

    Returns (model, optimizer, cfg, step).
    """
    import orbax.checkpoint as ocp

    cfg = load_config(save_dir)
    model = MlmEncoder(cfg, rngs=nnx.Rngs(0))
    model.train()
    optimizer = make_optimizer(model, schedule)

    mgr = checkpoint_manager(save_dir)
    step = mgr.latest_step() if step is None else step
    if step is None:
        raise RuntimeError(f"no checkpoints found in {save_dir}")
    restored = mgr.restore(step, args=ocp.args.Composite(
        model=ocp.args.StandardRestore(nnx.state(model, nnx.Param)),
        opt=ocp.args.StandardRestore(nnx.state(optimizer)),
    ))
    nnx.update(model, restored["model"])
    nnx.update(optimizer, restored["opt"])
    mgr.close()
    return model, optimizer, cfg, step
