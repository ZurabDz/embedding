"""Artifact locations and cross-stage defaults.

The single home for every path the pipeline stages share — each constant used
to be a string literal repeated across the scripts. Paths are CWD-relative on
purpose: the package imports from anywhere, but where artifacts land stays the
caller's choice (locally the geo_distill/ directory, on Kaggle /kaggle/working).
"""

DATA_SENTENCES = "data/sentences.txt"          # data -> tokenizer/teacher input

ARTIFACTS_DIR = "artifacts"
TOKENIZER_JSON = "artifacts/tokenizer.json"
TEACHER_EMB = "artifacts/teacher_emb.npy"
SENTENCES_JSON = "artifacts/sentences.json"
TEACHER_META = "artifacts/teacher_meta.json"
TEACHER_CACHE = "artifacts/teacher_cache.jsonl"
TEACHER_USAGE = "artifacts/teacher_usage.json"
MLM_CKPT_CACHE = "artifacts/mlm_checkpoint"    # per-repo Hub checkpoint cache

# Filenames inside a student --out-dir.
STUDENT_CONFIG = "student_config.json"
STUDENT_PARAMS = "student_params.msgpack"
TEACHER_MEAN = "teacher_mean.npy"

# The teacher model is part of the embedding cache key (teacher_cache.jsonl):
# changing it invalidates every cached vector, so it is a flag with this
# default rather than something to edit casually.
DEFAULT_TEACHER_MODEL = "gemini-embedding-2"
