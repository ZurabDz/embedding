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
STUDENT_PARAMS = "student_params.msgpack"   # the BEST epoch's parameters
TEACHER_MEAN = "teacher_mean.npy"
# Resume state, rewritten every --save-every epochs (and always at the end).
# Kept apart from STUDENT_PARAMS because they diverge the moment an epoch fails
# to improve: eval always wants the best parameters, a continuation always wants
# the latest ones.
STUDENT_STATE = "student_state.msgpack"     # latest params + Adam moments + rng counters
TRAIN_STATE = "train_state.json"            # epoch, best score, shuffle position, fingerprint

# The teacher model is part of the embedding cache key (teacher_cache.jsonl):
# changing it invalidates every cached vector, so it is a flag with this
# default rather than something to edit casually.
DEFAULT_TEACHER_MODEL = "gemini-embedding-2"

# local-teacher: fp16 shard cache root; each run settings combo gets its own
# subdirectory (see local_teacher.config_key), so stale vectors can't mix.
LOCAL_TEACHER_CACHE = "artifacts/local_teacher_cache"

# Like DEFAULT_TEACHER_MODEL, the model id is part of the cache config key:
# changing it re-embeds everything. 8B needs device_map="auto" over 2 GPUs
# (Kaggle T4x2); the 0.6B/4B variants fit one GPU (or CPU for smoke runs).
DEFAULT_LOCAL_TEACHER_MODEL = "Qwen/Qwen3-Embedding-8B"

# Where `local-teacher --push-to` lands and `fetch-teacher` pulls from by
# default: a private HF *dataset* repo holding the three teacher artifacts.
DEFAULT_TEACHER_DATASET_REPO = "ZurabDz/geo-teacher-qwen3-8b"

# The student's twin of the above, and a *model* repo rather than a dataset one
# (the two are separate Hub namespaces). `fetch-student` defaults to it;
# `train --push-to` does NOT — pushing needs a write token and replaces whatever
# the repo holds, so it stays opt-in.
DEFAULT_STUDENT_MODEL_REPO = "ZurabDz/ka-embed"
