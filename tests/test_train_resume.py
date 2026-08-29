"""Resumable training: a killed run continues instead of restarting.

The contract these pin is equivalence, not merely "it runs again": resuming has
to reproduce the uninterrupted run's next epochs exactly, which needs the Adam
moments (they carry the LR schedule's position), the dropout rng counters and
the numpy shuffle stream — parameters alone are not enough. Everything here is
offline and tiny; args come from the real parser so the tests cannot drift from
the flags run() actually receives.
"""
import json
import os

import numpy as np
import pytest

from geo_distill import config as paths
from geo_distill.cli import build_parser


# --------------------------------------------------------------------------- #
# Fixtures: a synthetic teacher + tokenizer laid out the way a real run finds it
# --------------------------------------------------------------------------- #
@pytest.fixture
def workdir(tmp_path, monkeypatch, bpe_tokenizer, sentences):
    """A CWD holding artifacts/{tokenizer.json,sentences.json,teacher_emb.npy}."""
    from geo_distill.teacher import run_synthetic

    monkeypatch.chdir(tmp_path)
    os.makedirs(paths.ARTIFACTS_DIR, exist_ok=True)
    bpe_tokenizer.save(paths.TOKENIZER_JSON)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(sentences), encoding="utf-8")
    run_synthetic(build_parser().parse_args(
        ["synthetic-teacher", "--input", str(corpus), "--dim", "16"]))
    return tmp_path


def train_args(*extra):
    return build_parser().parse_args([
        "train", "--epochs", "4", "--batch-size", "16", "--warmup", "5",
        "--dim", "32", "--depth", "1", "--heads", "2", "--mlp-dim", "64",
        "--max-len", "16", "--out-dir", "student", *extra])


def epoch_lines(capsys):
    return [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch ")]


# --------------------------------------------------------------------------- #
# The state itself
# --------------------------------------------------------------------------- #
def test_student_state_roundtrip(tmp_path, bpe_tokenizer):
    """Params, Adam moments and rng counters all survive; parameters alone do not."""
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from geo_distill.checkpoint import load_student_state, save_student_state
    from geo_distill.model import EmbeddingModel

    def build(seed):
        m = EmbeddingModel(vocab_size=bpe_tokenizer.get_vocab_size(), dim=32,
                           depth=1, heads=2, mlp_dim=64, max_len=16, out_dim=8,
                           dropout=0.1, rngs=nnx.Rngs(seed))
        m.train()
        sched = optax.warmup_cosine_decay_schedule(0.0, 1e-3, 2, 20, end_value=1e-4)
        return m, nnx.Optimizer(m, optax.adamw(sched, weight_decay=1e-2), wrt=nnx.Param)

    model, opt = build(0)
    tokens = jnp.asarray(np.random.RandomState(0).randint(0, 100, (8, 16)))
    mask = jnp.ones((8, 16), jnp.float32)

    @nnx.jit
    def step(model, optimizer):
        loss, grads = nnx.value_and_grad(
            lambda m: m(tokens, mask, deterministic=False).sum())(model)
        optimizer.update(model, grads)
        return loss

    for _ in range(5):
        step(model, opt)
    save_student_state(str(tmp_path), model, opt)

    # a *differently seeded* rebuild, so a no-op restore could not pass
    fresh, fresh_opt = build(7)
    before = nnx.to_pure_dict(nnx.state(fresh, nnx.Param))
    load_student_state(str(tmp_path), fresh, fresh_opt)

    for got, want in zip(nnx.to_flat_state(nnx.state(fresh, nnx.Param)),
                         nnx.to_flat_state(nnx.state(model, nnx.Param))):
        np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))
    assert not np.array_equal(before["proj"]["kernel"],
                              nnx.to_pure_dict(nnx.state(fresh, nnx.Param))["proj"]["kernel"])
    # the optimizer's step counts are what put the LR schedule back in place
    restored, original = nnx.state(fresh_opt), nnx.state(opt)
    assert int(restored["opt_state"][0]["count"][...]) == 5
    for got, want in zip(nnx.to_flat_state(restored), nnx.to_flat_state(original)):
        np.testing.assert_array_equal(np.asarray(got[1]), np.asarray(want[1]))
    # ...and the dropout counters, without which epoch 0's masks are replayed
    counts = nnx.to_flat_state(nnx.state(fresh, nnx.RngCount))
    assert counts and all(int(np.asarray(v)) == 5 for _, v in counts)


def test_train_state_is_plain_json(workdir, capsys):
    """fetch-student and any tooling must read a checkpoint's provenance
    without importing jax, so this half of it stays JSON."""
    from geo_distill.train import run

    run(train_args("--epochs", "1"))
    capsys.readouterr()
    state = json.loads((workdir / "student" / paths.TRAIN_STATE).read_text())
    assert state["epoch"] == 1 and state["epochs"] == 1
    assert state["numpy_rng"]["bit_generator"] == "PCG64"
    assert state["fingerprint"]["student_type"] == "scratch"
    assert isinstance(state["best_val_spearman"], float)


# --------------------------------------------------------------------------- #
# Equivalence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dropout", ["0.0", "0.1"])
def test_resume_reproduces_uninterrupted_run(workdir, monkeypatch, capsys, dropout):
    """The whole point: epochs 2..3 of a resumed run are the epochs the
    uninterrupted run would have produced, down to the loss digits."""
    from geo_distill import train as gt

    gt.run(train_args("--dropout", dropout))
    straight = epoch_lines(capsys)
    assert len(straight) == 4

    import shutil
    shutil.rmtree(workdir / "student")

    real = _kill_after(monkeypatch, gt, 2)
    with pytest.raises(KeyboardInterrupt):
        gt.run(train_args("--dropout", dropout))
    monkeypatch.setattr(gt, "save_train_state", real)
    capsys.readouterr()

    gt.run(train_args("--dropout", dropout, "--resume"))
    out = capsys.readouterr().out
    assert [l for l in out.splitlines() if l.startswith("epoch ")] == straight[2:]
    assert "resumed at epoch 2/4" in out


def _kill_after(monkeypatch, gt, epoch: int):
    """Simulate a session dying the instant epoch `epoch`'s checkpoint is
    complete on disk — the state a capped Kaggle run actually leaves behind."""
    real = gt.save_train_state

    def wrapper(out_dir, state):
        real(out_dir, state)
        if state["epoch"] == epoch:
            raise KeyboardInterrupt("session died")

    monkeypatch.setattr(gt, "save_train_state", wrapper)
    return real


def test_resume_keeps_the_best_epoch_not_the_latest(workdir, monkeypatch, capsys):
    """student_params.msgpack is the best epoch and student_state.msgpack the
    latest; a resume continues from the latter without ever regressing the
    former, even when the remaining epochs are worse."""
    from geo_distill.checkpoint import load_student
    from geo_distill import train as gt

    real = _kill_after(monkeypatch, gt, 2)
    with pytest.raises(KeyboardInterrupt):
        gt.run(train_args())
    monkeypatch.setattr(gt, "save_train_state", real)
    capsys.readouterr()

    from flax import serialization

    student = workdir / "student"
    best_before = json.loads(
        (student / paths.STUDENT_CONFIG).read_text())["best_val_spearman"]
    best_bytes = (student / paths.STUDENT_PARAMS).read_bytes()

    gt.run(train_args("--resume"))
    capsys.readouterr()
    cfg_after = json.loads((student / paths.STUDENT_CONFIG).read_text())
    assert cfg_after["best_val_spearman"] >= best_before

    # On this fixture epochs 2..3 do not beat epoch 1, so the best parameters
    # have to survive the resume byte for byte. Asserting on the score alone
    # would pass even if student_params.msgpack held the LATEST epoch, which is
    # the mistake this test exists to catch.
    assert (student / paths.STUDENT_PARAMS).read_bytes() == best_bytes
    # ...and they are genuinely not the latest ones the resume left behind
    latest = serialization.msgpack_restore(
        (student / paths.STUDENT_STATE).read_bytes())["params"]
    best = serialization.msgpack_restore(best_bytes)
    assert not np.array_equal(np.asarray(best["proj"]["kernel"]),
                              np.asarray(latest["proj"]["kernel"]))
    model, cfg, _ = load_student(str(student))
    assert cfg.best_val_spearman == cfg_after["best_val_spearman"]


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_resume_without_a_checkpoint_is_actionable(workdir):
    from geo_distill.train import run

    with pytest.raises(RuntimeError, match="without --resume"):
        run(train_args("--resume"))


def test_resume_past_the_end_asks_for_more_epochs(workdir, capsys):
    from geo_distill.train import run

    run(train_args("--epochs", "2"))
    capsys.readouterr()
    with pytest.raises(RuntimeError, match="raise it to continue"):
        run(train_args("--epochs", "2", "--resume"))


def test_raising_epochs_extends_a_finished_run(workdir, capsys):
    """--epochs is the one flag a resume may change: it is the total run
    length, so raising it re-stretches the cosine decay rather than erroring."""
    from geo_distill.train import run

    run(train_args("--epochs", "2"))
    capsys.readouterr()
    run(train_args("--epochs", "4", "--resume"))
    out = capsys.readouterr().out
    assert "re-stretched" in out                       # the LR change is announced
    assert [l[:9] for l in out.splitlines()
            if l.startswith("epoch ")] == ["epoch   2", "epoch   3"]


@pytest.mark.parametrize("flag,value,field", [
    (["--batch-size", "8"], None, "batch_size"),
    (["--dropout", "0.3"], None, "dropout"),
    (["--seed", "1"], None, "seed"),
    (["--sim-weight", "0.5"], None, "sim_weight"),
    (["--lr", "0.001"], None, "lr"),
    (["--dim", "64"], None, "dim"),
])
def test_fingerprint_guard_names_the_field(workdir, capsys, flag, value, field):
    """Anything that changes the training math must stop a resume, and say
    which setting moved — msgpack takes its shapes from the blob, so a silent
    restore would surface much later as an unrelated dimension error."""
    from geo_distill.train import run

    run(train_args("--epochs", "2"))
    capsys.readouterr()
    with pytest.raises(RuntimeError, match=field):
        run(train_args("--epochs", "4", "--resume", *flag))


def test_fingerprint_guard_catches_a_grown_corpus(workdir, capsys, sentences):
    """The corpus is not a flag — it grows underneath the run — and a longer
    one rescales the schedule and moves the target space."""
    from geo_distill.teacher import run_synthetic
    from geo_distill.train import run

    run(train_args("--epochs", "2"))
    capsys.readouterr()
    bigger = workdir / "bigger.txt"
    bigger.write_text("\n".join(sentences + [s + " მეტი" for s in sentences[:20]]),
                      encoding="utf-8")
    run_synthetic(build_parser().parse_args(
        ["synthetic-teacher", "--input", str(bigger), "--dim", "16"]))
    with pytest.raises(RuntimeError, match="corpus itself changed"):
        run(train_args("--epochs", "4", "--resume"))


def test_a_fresh_run_clears_stale_resume_state(workdir, capsys):
    """A directory holding one run's checkpoint beside another run's parameters
    is the one state --resume cannot detect, so a fresh run drops it up front."""
    from geo_distill.train import run

    run(train_args("--epochs", "2"))
    capsys.readouterr()
    stale = json.loads((workdir / "student" / paths.TRAIN_STATE).read_text())

    run(train_args("--epochs", "2", "--seed", "3"))   # different run, same dir
    capsys.readouterr()
    fresh = json.loads((workdir / "student" / paths.TRAIN_STATE).read_text())
    assert fresh["fingerprint"]["seed"] == 3 != stale["fingerprint"]["seed"]


def test_save_every_zero_still_checkpoints_at_the_end(workdir, capsys):
    from geo_distill.train import run

    run(train_args("--epochs", "2", "--save-every", "0"))
    capsys.readouterr()
    assert json.loads(
        (workdir / "student" / paths.TRAIN_STATE).read_text())["epoch"] == 2
