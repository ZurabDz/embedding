"""Student artifact IO: atomic writes, config/params save + load, and the
per-epoch resume state (latest params, Adam moments, rng and shuffle position).

flax and the students registry (which pulls in mlm -> jax) are imported lazily
inside the functions that need them, so the atomic-write helpers stay
importable from teacher.py, which is deliberately jax-free.
"""
from __future__ import annotations

import json
import os

import numpy as np

from geo_distill import config as paths


def atomic_write_bytes(path: str, data: bytes) -> None:
    """tmp + os.replace: a kill mid-write never tears the previous file, and a
    concurrently running reader never sees a half-written one."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_np_save(path: str, arr) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def save_student_config(out_dir: str, cfg: StudentConfig) -> None:
    atomic_write_text(os.path.join(out_dir, paths.STUDENT_CONFIG), cfg.to_json())


def save_student_params(out_dir: str, model) -> None:
    from flax import nnx, serialization

    data = serialization.to_bytes(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    atomic_write_bytes(os.path.join(out_dir, paths.STUDENT_PARAMS), data)


def load_student(model_dir: str):
    """Rebuild the student described by <model_dir>/student_config.json and
    restore its best parameters. Returns (model, cfg, spec)."""
    from flax import nnx, serialization

    from geo_distill.students import StudentConfig, spec_for

    cfg = StudentConfig.load(os.path.join(model_dir, paths.STUDENT_CONFIG))
    spec = spec_for(cfg.student_type)
    model = spec.rebuild(cfg)  # inference: dropout 0.0, whatever cfg.dropout says
    state = nnx.state(model, nnx.Param)  # template with the right structure/shapes
    with open(os.path.join(model_dir, paths.STUDENT_PARAMS), "rb") as f:
        pure = serialization.from_bytes(nnx.to_pure_dict(state), f.read())
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(model, state)
    return model, cfg, spec


# --------------------------------------------------------------------------- #
# Resume state
# --------------------------------------------------------------------------- #
def _resume_trees(model, optimizer):
    """The three trees a continuation has to restore, as (state, pure) pairs.

    Deliberately NOT the unfiltered nnx.state(model): that carries the RngKeys,
    which msgpack cannot encode (they are typed PRNG arrays) and whose tree
    shape changes with the dropout rate — MultiHeadAttention forks no stream at
    rate 0. The keys are reproducible from the seed anyway; only the *counts*
    move during training, and without them a resumed run replays the first
    epoch's dropout masks instead of continuing the stream.
    """
    from flax import nnx

    return {
        "params": nnx.state(model, nnx.Param),
        "rng": nnx.state(model, nnx.RngCount),
        "opt": nnx.state(optimizer),
    }


def save_student_state(out_dir: str, model, optimizer) -> None:
    """The LATEST parameters plus everything needed to continue from them.

    The optimizer state is where the Adam moments *and* the learning-rate
    schedule's position live (optax keeps a step count per chain element), so
    restoring it is what makes a resumed run pick the schedule up mid-decay
    instead of re-warming up from zero.
    """
    from flax import nnx, serialization

    trees = _resume_trees(model, optimizer)
    blob = serialization.to_bytes({k: nnx.to_pure_dict(v) for k, v in trees.items()})
    atomic_write_bytes(os.path.join(out_dir, paths.STUDENT_STATE), blob)


def load_student_state(out_dir: str, model, optimizer) -> None:
    """Restore save_student_state's blob into `model` and `optimizer` in place.

    msgpack carries no shapes of its own — from_bytes takes them from the blob
    and only checks that the *structure* matches — so a checkpoint from a
    differently sized run restores silently here and fails much later with an
    unrelated-looking dimension error. train.py's fingerprint guard runs first
    for exactly that reason.
    """
    from flax import nnx, serialization

    trees = _resume_trees(model, optimizer)
    template = {k: nnx.to_pure_dict(v) for k, v in trees.items()}
    with open(os.path.join(out_dir, paths.STUDENT_STATE), "rb") as f:
        pure = serialization.from_bytes(template, f.read())
    for key, state in trees.items():
        if not pure[key]:
            continue  # a student with no dropout stream has an empty rng tree
        nnx.replace_by_pure_dict(state, pure[key])
        nnx.update(optimizer if key == "opt" else model, state)


def save_train_state(out_dir: str, state: dict) -> None:
    """The plain-JSON half of a checkpoint: epoch, best score, the numpy shuffle
    position, the schedule, and the run fingerprint. Kept out of the msgpack so
    tooling (and `fetch-student`) can read a checkpoint's provenance without
    importing jax."""
    atomic_write_text(os.path.join(out_dir, paths.TRAIN_STATE),
                      json.dumps(state, indent=2) + "\n")


def load_train_state(out_dir: str) -> dict | None:
    path = os.path.join(out_dir, paths.TRAIN_STATE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
