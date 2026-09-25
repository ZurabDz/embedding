"""Place the Georgian student on the public Georgian embedding ladder.

Five labeled tasks, human-annotated (not teacher-derived), each with published
scores for 180+ models including this project's own teacher. Teacher-agreement
metrics say how faithfully the student COPIED Qwen3; these say whether the
embeddings are good for anything.

    # baselines only — no checkpoint needed, runs in ~2 min on a laptop
    python bench/georgian_bench.py

    # once you have a trained student
    python bench/georgian_bench.py --model-dir runs/student

Protocol notes that matter for comparability:
  * Classification is MTEB's 8-SHOT protocol: 10 runs, each undersampling the
    train split to 8 rows per label, LogisticRegression(max_iter=100). Fitting
    on the full train split gives a much prettier number that cannot be
    compared to anything published.
  * SIB-200 is pinned to the revision MTEB uses. On `main` the label column was
    renamed and the Georgian test split reads as 99 rows instead of 204.
  * This reimplements the protocols rather than importing `mteb`, which is
    close enough to rank yourself against the published table but is not
    bit-exact with the leaderboard. For an official number, `uv add mteb`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kaeval  # noqa: E402

SIB_REV = "a74d7350ea12af010cfb1c21e34f1f81fd2e615b"

# Published main_score values, pulled from github.com/embeddings-benchmark/results.
# The four Qwen3-Embedding-8B entries were verified against the raw JSON.
PUBLISHED = {
    "SIB200Clustering":     {"Qwen3-8B (your teacher)": 0.568, "jina-v3": 0.464,
                             "bge-m3": 0.370, "Qwen3-0.6B": 0.328, "LaBSE": 0.254,
                             "mE5-base": 0.228, "para-mMiniLM-L12": 0.186},
    "SIB200Classification": {"mE5-large-instruct": 0.818, "bge-m3": 0.715,
                             "LaBSE": 0.586, "para-mMiniLM-L12": 0.559,
                             "English-only floor": 0.149},
    "MassiveIntent":        {"Qwen3-8B (your teacher)": 0.703, "LaBSE": 0.483,
                             "para-mMiniLM-L12": 0.430, "mE5-small": 0.388,
                             "mE5-base": 0.376, "English-only floor": 0.013},
    "MassiveScenario":      {"Qwen3-8B (your teacher)": 0.782, "LaBSE": 0.534,
                             "mE5-base": 0.434},
    "GeorgianSentiment":    {},  # no published results exist for this dataset
}

# Lexical floors MEASURED on this machine with the baselines below, not guessed.
# Several published multilingual models sit BELOW these floors, which is why a
# bare score against the published table is not enough to claim anything.
LEXICAL_FLOOR = {
    "SIB200Clustering":     "v_measure 0.074 (lsa)",
    "SIB200Classification": "acc 0.405 (char-tfidf) / CV 0.692 (lsa)",
    "MassiveIntent":        "acc 0.466 (lsa)",
    "MassiveScenario":      "acc 0.534 (lsa)",
    "GeorgianSentiment":    "acc 0.600 / CV 0.820 (lsa)",
}

TARGETS = {
    # The discriminating task: character n-grams score 0.074, so anything well
    # above that is real semantics rather than surface form.
    "SIB200Clustering":     "MUST clear 0.074. good >=0.25 (=LaBSE), strong >=0.33 (=Qwen3-0.6B); RED FLAG <0.15",
    "SIB200Classification": "MUST clear 0.405 8-shot / 0.692 CV. good >=0.50; RED FLAG near 0.15 (cannot read Georgian)",
    # Weak discriminators: char n-grams already beat several published models here.
    "MassiveIntent":        "MUST clear 0.466 — char n-grams already beat mE5-base (0.376) and para-mMiniLM (0.430). Weak discriminator; also out-of-domain (6-token voice commands)",
    "MassiveScenario":      "MUST clear 0.534 — char n-grams already tie LaBSE (0.534). Weak discriminator; out-of-domain",
    "GeorgianSentiment":    "MUST clear 0.600 8-shot / 0.820 CV. Best domain match, but the lexical floor is high",
}


# --------------------------------------------------------------------------- #
# Task loading
# --------------------------------------------------------------------------- #
def _load(name, config=None, revision=None):
    from datasets import load_dataset
    kw = {}
    if revision:
        kw["revision"] = revision
    return load_dataset(name, config, **kw) if config else load_dataset(name, **kw)


def load_tasks(which):
    """-> {task_name: dict(kind, train=(texts,labels), test=(texts,labels))}"""
    out = {}
    if "SIB200Clustering" in which or "SIB200Classification" in which:
        d = _load("mteb/sib200", "kat_Geor", SIB_REV)
        pool_t = list(d["train"]["text"]) + list(d["validation"]["text"]) + list(d["test"]["text"])
        pool_y = (list(d["train"]["category"]) + list(d["validation"]["category"])
                  + list(d["test"]["category"]))
        if "SIB200Clustering" in which:
            out["SIB200Clustering"] = {"kind": "clustering", "pool": (pool_t, pool_y)}
        if "SIB200Classification" in which:
            out["SIB200Classification"] = {
                "kind": "classification",
                "train": (list(d["train"]["text"]), list(d["train"]["category"])),
                "test": (list(d["test"]["text"]), list(d["test"]["category"])),
                "pool": (pool_t, pool_y),
            }
    for key, repo in [("MassiveIntent", "mteb/amazon_massive_intent"),
                      ("MassiveScenario", "mteb/amazon_massive_scenario")]:
        if key in which:
            d = _load(repo, "ka")
            out[key] = {
                "kind": "classification",
                "train": (list(d["train"]["text"]), list(d["train"]["label"])),
                "test": (list(d["test"]["text"]), list(d["test"]["label"])),
                "pool": None,  # 11.5k train rows: skip the CV variant, 8-shot is the standard
            }
    if "GeorgianSentiment" in which:
        d = _load("asparius/Georgian-Sentiment")
        out["GeorgianSentiment"] = {
            "kind": "classification",
            "train": (list(d["train"]["text"]), list(d["train"]["label"])),
            "test": (list(d["test"]["text"]), list(d["test"]["label"])),
            "pool": (list(d["train"]["text"]) + list(d["test"]["text"]),
                     list(d["train"]["label"]) + list(d["test"]["label"])),
        }
    return out


# --------------------------------------------------------------------------- #
# Protocols
# --------------------------------------------------------------------------- #
def undersample(labels, per_label: int, seed: int):
    """MTEB's few-shot draw: `per_label` train rows for each class."""
    rng = np.random.default_rng(seed)
    y = np.asarray(labels)
    idx = []
    for lab in np.unique(y):
        pool = np.flatnonzero(y == lab)
        idx.extend(rng.choice(pool, min(per_label, len(pool)), replace=False))
    return np.array(sorted(idx))


def run_classification(embed, task, n_runs=10, per_label=8, seed=0, cache=None):
    """MTEB 8-shot linear probe. Returns aggregate stats + per-test-item accuracy
    (averaged over runs) so two models can be compared with a paired bootstrap.

    Train and test are embedded in ONE call: a TfidfVectorizer fit separately on
    each would put them in different feature spaces. That does mean the lexical
    baseline sees test vocabulary when building its features, which makes the
    floor slightly stronger than it would otherwise be — the conservative
    direction for a bar the student has to clear.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    tr_x, tr_y = task["train"]
    te_x, te_y = task["test"]

    # Draw every few-shot sample up front, then embed only the union. Avoids
    # embedding 11.5k MASSIVE train rows to use ~4k of them.
    draws = [undersample(tr_y, per_label, seed + i) for i in range(n_runs)]
    need = np.unique(np.concatenate(draws))
    pos = {orig: k for k, orig in enumerate(need)}

    X = _embed_cached(embed, [tr_x[i] for i in need] + list(te_x),
                      cache, f"{task['_name']}:trainsub+test")
    Xtr, Xte = X[:len(need)], X[len(need):]
    ytr_all, yte = np.asarray(tr_y), np.asarray(te_y)

    accs, f1s, correct = [], [], np.zeros(len(te_y))
    for i, d in enumerate(draws):
        rows = [pos[j] for j in d]
        clf = LogisticRegression(max_iter=100, random_state=seed + i)
        clf.fit(Xtr[rows], ytr_all[d])
        pred = clf.predict(Xte)
        hit = (pred == yte).astype(float)
        correct += hit
        accs.append(hit.mean())
        f1s.append(f1_score(yte, pred, average="macro", zero_division=0))
    correct /= n_runs

    mean, lo, hi = kaeval.bootstrap_ci(correct, seed=seed)
    return {"accuracy": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "ci95": [lo, hi],
            "n_test": len(te_y), "n_labels": int(len(np.unique(ytr_all))),
            "_per_item": correct}


def run_classification_cv(embed, task, folds=5, seed=0, cache=None):
    """Higher-power secondary number. 8-shot on a 2-class, 330-row train split is
    extremely noisy; this uses all pooled rows. Not leaderboard-comparable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    if not task.get("pool"):
        return None
    x, y = task["pool"]
    X = _embed_cached(embed, x, cache, f"{task['_name']}:pool")
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    s = cross_val_score(LogisticRegression(C=1.0, max_iter=1000), X, np.asarray(y),
                        cv=cv, scoring="accuracy")
    return {"cv_accuracy": float(s.mean()), "cv_sd": float(s.std()), "n": len(y)}


def run_clustering(embed, task, n_runs=10, seed=0, cache=None):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, v_measure_score

    x, y = task["pool"]
    X = _embed_cached(embed, x, cache, f"{task['_name']}:pool")
    y = np.asarray(y)
    k = len(np.unique(y))
    rng = np.random.default_rng(seed)
    v, ari = [], []
    for i in range(n_runs):
        pick = rng.integers(0, len(y), len(y))  # bootstrap resample
        km = MiniBatchKMeans(n_clusters=k, batch_size=500, n_init=3, random_state=seed + i)
        lab = km.fit_predict(X[pick])
        v.append(v_measure_score(y[pick], lab))
        ari.append(adjusted_rand_score(y[pick], lab))
    return {"v_measure": float(np.mean(v)), "v_measure_sd": float(np.std(v)),
            "ari": float(np.mean(ari)), "n": len(y), "n_clusters": k}


# --------------------------------------------------------------------------- #
def _embed_cached(embed, texts, cache, key):
    if cache is None:
        return embed(texts)
    ck = (embed.name, key, len(texts))
    if ck not in cache:
        cache[ck] = embed(texts)
    return cache[ck]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", help="trained student dir (student_config.json + params). "
                                       "Omit to run baselines only.")
    p.add_argument("--max-len", type=int, default=None,
                   help="override the student's max_len (for a length sweep)")
    p.add_argument("--tasks", default=",".join(TARGETS),
                   help="comma-separated subset of: " + ",".join(TARGETS))
    p.add_argument("--no-baselines", action="store_true",
                   help="skip the lexical floors (NOT recommended — they are the bar)")
    p.add_argument("--random-floor", action="store_true",
                   help="also run the semantics-free random-projection floor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="bench/results.json")
    args = p.parse_args()

    which = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = set(which) - set(TARGETS)
    if unknown:
        p.error(f"unknown task(s): {sorted(unknown)}; choose from {sorted(TARGETS)}")

    embedders = []
    if args.model_dir:
        embedders.append(kaeval.student_embedder(args.model_dir, max_len=args.max_len))
    if not args.no_baselines:
        embedders.append(kaeval.tfidf_embedder("char_wb", (3, 5)))
        embedders.append(kaeval.lsa_embedder(300, "char_wb", (3, 5)))
        embedders.append(kaeval.tfidf_embedder("word", (1, 1)))
    if args.random_floor:
        embedders.append(kaeval.random_embedder())
    if not embedders:
        p.error("nothing to evaluate: pass --model-dir, or drop --no-baselines")

    print(f"loading tasks: {', '.join(which)}")
    tasks = load_tasks(which)
    for name, t in tasks.items():
        t["_name"] = name

    results = {}
    for name, task in tasks.items():
        print(f"\n{'=' * 76}\n{name}   ({TARGETS[name]})\n{'=' * 76}")
        results[name] = {}
        per_item = {}
        for embed in embedders:
            cache = {}
            if task["kind"] == "clustering":
                r = run_clustering(embed, task, seed=args.seed, cache=cache)
                print(f"  {embed.name:34s} v_measure {r['v_measure']:.3f} "
                      f"(sd {r['v_measure_sd']:.3f})  ARI {r['ari']:.3f}")
            else:
                r = run_classification(embed, task, seed=args.seed, cache=cache)
                per_item[embed.name] = r.pop("_per_item")
                cv = run_classification_cv(embed, task, seed=args.seed, cache=cache)
                if cv:
                    r.update(cv)
                line = (f"  {embed.name:34s} acc {r['accuracy']:.3f} "
                        f"(sd {r['accuracy_sd']:.3f}, 95% CI [{r['ci95'][0]:.3f},"
                        f"{r['ci95'][1]:.3f}])  macroF1 {r['macro_f1']:.3f}")
                if cv:
                    line += f"  |  5-fold CV {r['cv_accuracy']:.3f}"
                print(line)
            results[name][embed.name] = r

        # The number that actually matters: student minus the STRONGEST lexical
        # floor. Comparing against a deliberately weak baseline flatters the model.
        lexical = [n for n in per_item if n.startswith(("tfidf[", "lsa["))]
        floor = max(lexical, key=lambda n: per_item[n].mean()) if lexical else None
        stu = next((e.name for e in embedders if e.name.startswith("student")), None)
        if floor and stu and stu in per_item:
            d = kaeval.paired_delta(per_item[stu], per_item[floor], seed=args.seed)
            verdict = "SIGNIFICANT" if d["significant"] else "NOT significant"
            print(f"  -> student - {floor} = {d['delta']:+.3f} "
                  f"[{d['lo']:+.3f}, {d['hi']:+.3f}]  ({verdict})")
            results[name]["_paired_delta_vs_lexical_floor"] = {"floor": floor, **d}

        print(f"  measured lexical floor: {LEXICAL_FLOOR[name]}")
        if PUBLISHED.get(name):
            ref = "  ".join(f"{k} {v:.3f}" for k, v in PUBLISHED[name].items())
            print(f"  published: {ref}")
        else:
            print("  published: none exist for this dataset — the baselines above are the reference")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
