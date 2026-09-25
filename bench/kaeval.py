"""Shared pieces for evaluating the Georgian student embedder.

Every benchmark below needs the same three things: embed with the student,
embed with a dumb lexical baseline, and compare the two with a paired
confidence interval. Nothing here imports jax unless you actually ask for the
student, so the baselines run on a laptop with no ML stack warm.

The lexical baselines are not decoration. Character n-grams are a strong
sentence representation for an agglutinative language written in one script,
and a student that cannot beat them did not learn anything worth shipping.
"""
from __future__ import annotations

import os

import numpy as np

# --------------------------------------------------------------------------- #
# Embedders. Each is a callable: list[str] -> (N, D) matrix, rows L2-normalized.
# --------------------------------------------------------------------------- #


def student_embedder(model_dir: str, max_len: int | None = None, batch: int = 256):
    """The distilled student. Imports jax lazily — only pay for it if used."""
    from geo_distill.checkpoint import load_student
    from geo_distill.data import load_tokenizer
    from geo_distill.model import embed_in_batches

    model, cfg, spec = load_student(model_dir)
    tok = load_tokenizer(cfg.tokenizer)
    spec.validate_tokenizer(tok)
    L = max_len or cfg.max_len

    def embed(sentences):
        tokens, mask = spec.encode(tok, list(sentences), L)
        return embed_in_batches(model, tokens, mask, batch=batch)

    embed.name = f"student[{os.path.basename(model_dir.rstrip('/'))}@{L}]"
    embed.cfg = cfg
    return embed


def tfidf_embedder(analyzer: str = "char_wb", ngram=(3, 5), min_df: int = 2):
    """The lexical floor. `char_wb` 3-5 grams is the bar that actually matters
    for Georgian; word unigrams are the easier, weaker control."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    def embed(sentences):
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram,
                              min_df=min_df, sublinear_tf=True, norm="l2")
        return vec.fit_transform(list(sentences))  # sparse; sklearn handles it

    embed.name = f"tfidf[{analyzer}{ngram}]"
    return embed


def lsa_embedder(dim: int = 300, analyzer: str = "char_wb", ngram=(3, 5), seed: int = 0):
    """TF-IDF followed by truncated SVD — i.e. classic LSA.

    This is the honest lexical floor for CLUSTERING. Raw sparse TF-IDF is a
    straw man for k-means, which degrades badly in tens of thousands of sparse
    dimensions, so beating it proves nothing. Reducing to a few hundred dense
    dimensions first is what a competent engineer would actually do without any
    neural model, and it is a far harder bar.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    def embed(sentences):
        sentences = list(sentences)
        X = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2,
                            sublinear_tf=True, norm="l2").fit_transform(sentences)
        k = min(dim, min(X.shape) - 1)
        Z = TruncatedSVD(k, random_state=seed).fit_transform(X)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    embed.name = f"lsa[{analyzer}->{dim}d]"
    return embed


def random_embedder(dim: int = 1024, seed: int = 0):
    """Deterministic hashed random projection of character trigrams.

    The genuine floor: this has no semantics at all, only surface form. Also
    doubles as a smoke test for the whole harness when no checkpoint exists yet.
    """
    def embed(sentences):
        rng = np.random.default_rng(seed)
        proj_cache: dict[int, np.ndarray] = {}
        out = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            s = f" {s.strip()} "
            for n in (3,):
                for j in range(len(s) - n + 1):
                    h = hash(s[j:j + n]) % 65536
                    if h not in proj_cache:
                        proj_cache[h] = np.random.default_rng(seed + h).normal(size=dim)
                    out[i] += proj_cache[h]
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out

    embed.name = f"random-proj[{dim}]"
    return embed


# --------------------------------------------------------------------------- #
# Statistics. Point estimates without intervals are how people fool themselves.
# --------------------------------------------------------------------------- #


def bootstrap_ci(per_item, n_boot: int = 2000, seed: int = 0, groups=None):
    """95% CI for the mean of per-item scores.

    `groups` (article id, question id, ...) resamples whole clusters, because
    items inside one cluster are correlated and resampling them independently
    produces a dishonestly narrow interval.
    """
    x = np.asarray(per_item, dtype=float)
    rng = np.random.default_rng(seed)
    units = ([np.flatnonzero(np.asarray(groups) == k) for k in np.unique(groups)]
             if groups is not None else None)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        if units is None:
            boot[i] = x[rng.integers(0, len(x), len(x))].mean()
        else:
            pick = rng.integers(0, len(units), len(units))
            boot[i] = x[np.concatenate([units[j] for j in pick])].mean()
    return float(x.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_delta(per_item_a, per_item_b, n_boot: int = 2000, seed: int = 0, groups=None):
    """CI on the DIFFERENCE between two models scored on the SAME items.

    Two separate confidence intervals that happen to overlap do NOT mean the
    models are indistinguishable — the comparison has to be paired.
    """
    a, b = np.asarray(per_item_a, float), np.asarray(per_item_b, float)
    assert a.shape == b.shape, "paired comparison needs the same items"
    mean, lo, hi = bootstrap_ci(a - b, n_boot=n_boot, seed=seed, groups=groups)
    return {"delta": mean, "lo": lo, "hi": hi, "significant": lo > 0 or hi < 0}


# --------------------------------------------------------------------------- #
# Geometry. Catches a collapsed space that task accuracy can still hide.
# --------------------------------------------------------------------------- #


def geometry(emb, sample: int = 4000, seed: int = 0):
    """Thresholds must be read against the teacher measured the SAME way, not
    against numbers from papers about uncentered English encoders. On this
    repo's own centered teacher: participation ratio ~113, effective rank ~292,
    mean pairwise cosine ~0. The student's structural ceiling is the rank-384
    teacher (effective rank ~191), because 1024 dims exit a 384-wide encoder.
    """
    e = np.asarray(emb, dtype=np.float64)
    e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
    rng = np.random.default_rng(seed)
    if len(e) > sample:
        e = e[rng.choice(len(e), sample, replace=False)]
    n = len(e)
    sim = e @ e.T
    mean_cos = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    sv = np.linalg.svd(e - e.mean(0, keepdims=True), compute_uv=False)
    p = sv ** 2 / (sv ** 2).sum()
    np.fill_diagonal(sim, -np.inf)
    occ = np.bincount(sim.argmax(1), minlength=n)
    return {
        "mean_cosine": float(mean_cos),
        "cosine_sd": float(sim[np.triu_indices(n, 1)][np.isfinite(sim[np.triu_indices(n, 1)])].std()),
        "effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum())),
        "participation_ratio": float(1.0 / (p ** 2).sum()),
        "hub_skew": float(((occ - occ.mean()) ** 3).mean() / (occ.std() ** 3 + 1e-12)),
    }
