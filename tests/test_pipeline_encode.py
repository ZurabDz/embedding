"""Micro-batch planner + thread-pipeline scheduler, torch-free.

These are the pure-Python halves of local_teacher.make_encoder's fast path:
_plan_micro_batches packs length-sorted indices under a token budget, and
_run_groups executes groups (concurrently on a multi-GPU split) with the
split-and-requeue OOM behavior. Everything here runs with fake runners, so
the suite needs no torch, GPU, or model download — the model-side thread
safety preconditions (use_cache=False, per-thread inference_mode) are
documented in local_teacher.py and validated on the GPU box.
"""
import threading

import numpy as np
import pytest

from geo_distill.local_teacher import (_assemble, _clamp_budget,
                                       _plan_micro_batches, _run_groups)

DIM = 8


def _rows(group):
    """Deterministic row per index, independent of grouping/thread order."""
    return np.stack([np.full(DIM, i, dtype=np.float32) for i in group])


# ---------------------------------------------------------------------------
# _plan_micro_batches
# ---------------------------------------------------------------------------


def test_planner_covers_all_indices_in_order_within_budget():
    lengths = sorted([3, 5, 8, 8, 13, 21, 34, 55, 89])
    groups = _plan_micro_batches(lengths, budget=100)

    assert [i for g in groups for i in g] == list(range(len(lengths)))
    for g in groups:
        # ascending input -> the group's last length is its padded width
        assert len(g) * lengths[g[-1]] <= 100


def test_planner_singleton_when_budget_tiny():
    # every sequence exceeds the budget on its own -> one group each
    assert _plan_micro_batches([50, 60, 70], budget=10) == [[0], [1], [2]]


def test_planner_single_group_when_budget_ample():
    assert _plan_micro_batches([2, 3, 4], budget=1000) == [[0, 1, 2]]


def test_planner_empty():
    assert _plan_micro_batches([], budget=100) == []


# ---------------------------------------------------------------------------
# _run_groups
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_threads", [1, 2, 3])
def test_threaded_equals_sequential(n_threads):
    groups = _plan_micro_batches(list(range(1, 65)), budget=64)
    parts = _run_groups(groups, _rows, n_threads, on_oom=lambda g: None)
    np.testing.assert_array_equal(_assemble(parts, 64), _rows(list(range(64))))


def test_oom_splits_and_requeues_until_it_fits():
    oom_calls = []

    def runner(group):
        if len(group) > 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return _rows(group)

    for n_threads in (1, 3):
        oom_calls.clear()
        parts = _run_groups([list(range(32))], runner, n_threads,
                            on_oom=oom_calls.append)
        assert oom_calls  # the budget-shrinking hook fired
        assert max(len(g) for g, _ in parts) <= 4
        np.testing.assert_array_equal(_assemble(parts, 32),
                                      _rows(list(range(32))))


def test_oom_on_single_index_becomes_systemexit():
    def runner(group):
        raise RuntimeError("CUDA out of memory")

    for n_threads in (1, 2):
        with pytest.raises(SystemExit, match="single sentence"):
            _run_groups([[0], [1]], runner, n_threads, on_oom=lambda g: None)


def test_non_oom_error_propagates_without_hanging():
    def runner(group):
        raise ValueError("boom")

    before = threading.active_count()
    for n_threads in (1, 3):
        with pytest.raises(ValueError, match="boom"):
            _run_groups([[0], [1], [2]], runner, n_threads,
                        on_oom=lambda g: None)
    # _run_groups joins its workers before re-raising: none left behind
    assert threading.active_count() == before


def test_systemexit_from_worker_reraises_in_caller():
    """SystemExit is not an Exception — a worker thread must still trap it,
    or the thread dies silently and rows are left unfilled."""
    def runner(group):
        raise SystemExit("config error")

    for n_threads in (1, 3):
        with pytest.raises(SystemExit, match="config error"):
            _run_groups([[0], [1]], runner, n_threads, on_oom=lambda g: None)


def test_on_oom_hook_failure_is_loud_not_silent():
    """A crash inside the OOM handler (e.g. print -> BlockingIOError on a
    notebook pipe) must abort the run — never return partial coverage that
    would be scattered into uninitialized rows and cached forever."""
    def runner(group):
        if len(group) > 1:
            raise RuntimeError("CUDA out of memory")
        return _rows(group)

    def bad_hook(group):
        raise BlockingIOError("stdout pipe full")

    for n_threads in (1, 3):
        with pytest.raises(BlockingIOError):
            _run_groups([list(range(8))], runner, n_threads, on_oom=bad_hook)


def test_assemble_refuses_partial_coverage():
    parts = [([0, 1], _rows([0, 1]))]  # rows 2..3 missing
    with pytest.raises(RuntimeError, match="covered 2/4"):
        _assemble(parts, 4)
    with pytest.raises(RuntimeError, match="covered 0/1"):
        _assemble([], 1)


def test_budget_clamp_does_not_cascade():
    """One overload episode must settle the budget at half the failing cost,
    not halve once per already-planned group all the way to the floor."""
    b = [16384]
    assert _clamp_budget(b, cost=16384, floor=512) and b[0] == 8192
    for _ in range(7):  # the other stale-planned groups OOM at the same cost
        assert not _clamp_budget(b, cost=16384, floor=512)
    assert b[0] == 8192
    # a requeued half that still OOMs steps down one more level
    assert _clamp_budget(b, cost=8192, floor=512) and b[0] == 4096
    # the floor holds
    _clamp_budget(b, cost=700, floor=512)
    assert b[0] == 512
    assert not _clamp_budget(b, cost=700, floor=512) and b[0] == 512


def test_torch_ops_threaded_equals_sequential():
    """Real tensor math through the scheduler: grouping/threading must never
    change the values (tiny CPU matmul — safe for a laptop-only run)."""
    torch = pytest.importorskip("torch")

    proj = torch.linspace(-1, 1, DIM * DIM).reshape(DIM, DIM)

    def runner(group):
        x = torch.stack([torch.full((DIM,), float(i)) for i in group])
        with torch.inference_mode():
            return (x @ proj).numpy().copy()

    groups = _plan_micro_batches([1] * 40, budget=7)
    seq = _assemble(_run_groups(groups, runner, 1, on_oom=lambda g: None), 40)
    par = _assemble(_run_groups(groups, runner, 3, on_oom=lambda g: None), 40)
    np.testing.assert_array_equal(seq, par)
    assert not np.isnan(seq).any()
