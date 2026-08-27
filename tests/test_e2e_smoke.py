"""End-to-end pipeline smoke via the real CLI, in a scratch CWD.

Marked slow (excluded by default); run with:  uv run pytest -m slow
"""
import json
import pathlib
import subprocess
import sys

import pytest

SAMPLES = pathlib.Path(__file__).parent / "data" / "sample_sentences.txt"

pytestmark = pytest.mark.slow


def cli(cwd, *argv, check=True):
    return subprocess.run([sys.executable, "-m", "geo_distill", *argv],
                          cwd=cwd, check=check, capture_output=True, text=True)


def test_scratch_pipeline(tmp_path):
    cli(tmp_path, "tokenizer", "--input", str(SAMPLES),
        "--vocab-size", "800", "--min-frequency", "1")
    cli(tmp_path, "synthetic-teacher", "--input", str(SAMPLES))
    r = cli(tmp_path, "train", "--epochs", "2", "--batch-size", "16",
            "--warmup", "5", "--out-dir", "student")
    assert "Best val Spearman" in r.stdout
    assert (tmp_path / "student" / "teacher_mean.npy").is_file()
    cfg = json.loads((tmp_path / "student" / "student_config.json").read_text())
    assert cfg["student_type"] == "scratch" and cfg["center"] is True

    r = cli(tmp_path, "eval", "--model-dir", "student",
            "--query", "ქართული ღვინო")
    assert "Spearman" in r.stdout and "Query:" in r.stdout


def test_mlm_pipeline_and_mispair_guard(tmp_path, tiny_checkpoint):
    ck_dir, _ = tiny_checkpoint
    # the checkpoint's tokenizer twin: same recipe, same corpus, same vocab
    cli(tmp_path, "tokenizer", "--input", str(SAMPLES),
        "--vocab-size", "600", "--min-frequency", "1")
    cli(tmp_path, "synthetic-teacher", "--input", str(SAMPLES))
    r = cli(tmp_path, "train", "--mlm-checkpoint", str(ck_dir),
            "--tokenizer", "artifacts/tokenizer.json",
            "--epochs", "1", "--batch-size", "8", "--max-len", "32",
            "--warmup", "2", "--out-dir", "mlm_student")
    assert "loaded MLM encoder" in r.stdout

    raw = (tmp_path / "mlm_student" / "student_params.msgpack").read_bytes()
    assert b"head" not in raw  # dead MLM head stripped from the saved tree

    r = cli(tmp_path, "eval", "--model-dir", "mlm_student")
    assert "Spearman" in r.stdout

    # mispairing the old 2-special recipe with the encoder is a HARD error
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    bad = Tokenizer(models.BPE(unk_token="[UNK]"))
    bad.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(
        vocab_size=600, min_frequency=1, special_tokens=["[PAD]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    bad.train_from_iterator(SAMPLES.read_text().splitlines(), trainer)
    bad.save(str(tmp_path / "bad_tok.json"))
    r = cli(tmp_path, "train", "--mlm-checkpoint", str(ck_dir),
            "--tokenizer", "bad_tok.json", "--epochs", "1",
            "--out-dir", "mispair", check=False)
    assert r.returncode != 0
    assert "special tokens" in r.stderr
