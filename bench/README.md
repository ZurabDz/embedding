# Georgian embedding benchmark

Places the distilled student on the **public Georgian embedding ladder**, where
180+ models — including this project's own teacher — already have published
scores.

## Why this exists

The training log reports Pearson / Spearman / top-1-NN agreement with the
teacher. That measures how faithfully the student *copied* Qwen3-Embedding-8B.
It is maximised by a student that copies the teacher's mistakes too, and it says
nothing about whether Georgian sentences with the same meaning end up near each
other. These tasks use **human labels**, so they can.

## Running it

```bash
uv pip install scikit-learn            # not yet in pyproject.toml

python bench/georgian_bench.py                        # baselines only, ~30 s
python bench/georgian_bench.py --model-dir runs/student
python bench/georgian_bench.py --model-dir runs/student --tasks SIB200Clustering
python bench/georgian_bench.py --model-dir runs/student --max-len 128   # length sweep
```

Everything runs on CPU. The full suite with a 34M student takes about a minute
on a MacBook Air; datasets total under 20k short sentences and are cached by
`datasets` after the first run.

## The scoreboard

Lexical floors were **measured on this machine**, not guessed. Published values
come from `github.com/embeddings-benchmark/results`; the four Qwen3-8B entries
were verified against the raw JSON.

| Task | char n-gram floor | LaBSE (471M) | mE5-base | Teacher (Qwen3-8B) |
|---|---|---|---|---|
| SIB200 clustering (v-measure) | **0.074** | 0.254 | 0.228 | 0.568 |
| SIB200 classification (8-shot acc) | **0.405** | 0.586 | — | — |
| MASSIVE intent (8-shot acc) | **0.466** | 0.483 | 0.376 | 0.703 |
| MASSIVE scenario (8-shot acc) | **0.534** | 0.534 | 0.434 | 0.782 |
| Georgian sentiment (5-fold CV acc) | **0.820** | — | — | — |

### Read this before interpreting any score

**SIB-200 clustering is the task that matters.** Character n-grams get 0.074
there, so it cannot be solved by surface form — a good score is real semantics.
Clearing 0.25 means matching LaBSE, a 471M multilingual model, with 34M
monolingual parameters.

**The MASSIVE tasks are weak discriminators.** Character n-grams score 0.466 and
0.534, which already beats multilingual-e5-base (0.376 / 0.434) and ties LaBSE
on scenario. A student scoring 0.45 on intent looks respectable next to the
published table and is in fact *worse than TF-IDF*. Always read the
`student - lexical floor` line, not the raw number. MASSIVE is also
out-of-distribution: 6-token voice-assistant commands versus the ~19-token news
sentences the student trained on, so a low score there alongside a healthy
SIB-200 score means domain shift, not a broken model.

**Georgian sentiment has the highest floor.** TF-IDF reaches 0.820 under 5-fold
CV, so the model has to beat that to have contributed anything.

## Protocol notes

Classification uses MTEB's **8-shot** protocol: ten runs, each undersampling the
train split to 8 rows per label, then `LogisticRegression(max_iter=100)`.
Fitting on the full train split produces a much prettier number that cannot be
compared to anything published. A 5-fold CV number is reported alongside for the
small datasets, where 8-shot is very noisy; it is higher-powered but *not*
leaderboard-comparable.

SIB-200 is pinned to revision `a74d7350ea12af010cfb1c21e34f1f81fd2e615b`. On
`main` the label column was renamed and the Georgian test split reads as 99 rows
instead of 204.

These are reimplementations of the MTEB protocols, close enough to rank yourself
against the published table but not bit-exact with the leaderboard. For an
official submission, `uv add mteb` and wrap the student in its encoder API.

Confidence intervals matter here. SIB-200's test split is 204 rows, giving a
bootstrap CI of roughly ±0.06 — differences smaller than that are noise.

## Files

- `kaeval.py` — embedders (student, char/word TF-IDF, LSA, random-projection
  floor), bootstrap and paired-delta statistics, and the geometry/collapse panel.
- `georgian_bench.py` — the five tasks and their protocols.

`geometry()` in `kaeval.py` detects a collapsed embedding space, which task
accuracy can hide. Read its thresholds against this repo's own centered teacher
(participation ratio ~113, effective rank ~292, mean pairwise cosine ~0), not
against numbers from papers on uncentered English encoders. The student's
structural ceiling is the rank-384 teacher — effective rank ~191 — because 1024
output dimensions leave a 384-wide encoder.
