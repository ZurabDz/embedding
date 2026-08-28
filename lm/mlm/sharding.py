"""Device placement: single-host data parallelism and host->device streaming."""

import jax
from flax import nnx
import jax.numpy as jnp


def _auto_mesh(n: int):
    """A mesh whose axes stay in GSPMD 'auto' mode.

    jax >= 0.11 builds *explicit*-mode meshes by default, and under those the
    embedding gather raises ShardingTypeError the moment the batch is sharded:
    it refuses to infer an output sharding for a sharded-index gather. This
    pipeline is plain data parallelism and wants GSPMD inference everywhere,
    so ask for Auto axes — and fall back for older jax without axis_types.
    """
    try:
        return jax.make_mesh((n,), ("data",),
                             axis_types=(jax.sharding.AxisType.Auto,))
    except (TypeError, AttributeError):
        return jax.make_mesh((n,), ("data",))


def make_shardings(batch_size: int):
    """Single-host data parallelism: batch split over devices, params replicated.

    Returns (data_sharding, replicated) — or (None, None) on one device, so the
    single-accelerator path stays free of any sharding machinery.
    """
    n = jax.device_count()
    if n == 1:
        return None, None
    if batch_size % n:
        raise RuntimeError(
            f"--batch-size {batch_size} is not divisible by {n} devices; "
            f"uneven sharding fails deep inside pjit with a much worse message"
        )
    mesh = _auto_mesh(n)
    return (jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data", None)),
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()))


def replicate(model, optimizer, repl) -> None:
    """Commit params and optimizer state to every device, once, up front.

    Sharding only the batch would leave these uncommitted on one device; GSPMD
    then infers replication and returns multi-device-committed outputs, so step
    two sees different input shardings than step one and recompiles.
    """
    if repl is None:
        return
    nnx.update(model, jax.device_put(nnx.state(model), repl))
    nnx.update(optimizer, jax.device_put(nnx.state(optimizer), repl))


def to_device(batch: dict, sharding):
    """Host numpy -> device arrays, sharded along the batch axis when asked."""
    if sharding is None:
        return {k: jnp.asarray(v) for k, v in batch.items()}
    return {k: jax.device_put(v, sharding) for k, v in batch.items()}


def device_stream(iter_ds, sharding):
    """Overlap host masking with device compute, and double-buffer on device."""
    import grain.experimental

    target = sharding if sharding is not None else jax.devices()[0]
    return grain.experimental.device_put(
        iter_ds, target, cpu_buffer_size=4, device_buffer_size=2)
