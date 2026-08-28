"""Student artifact IO: atomic writes and config/params save + load.

flax and the students registry (which pulls in mlm -> jax) are imported lazily
inside the functions that need them, so the atomic-write helpers stay
importable from teacher.py, which is deliberately jax-free.
"""
from __future__ import annotations

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
    model = spec.rebuild(cfg)
    state = nnx.state(model, nnx.Param)  # template with the right structure/shapes
    with open(os.path.join(model_dir, paths.STUDENT_PARAMS), "rb") as f:
        pure = serialization.from_bytes(nnx.to_pure_dict(state), f.read())
    nnx.replace_by_pure_dict(state, pure)
    nnx.update(model, state)
    return model, cfg, spec
