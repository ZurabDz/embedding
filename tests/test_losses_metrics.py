"""Losses at fixed points, metrics against scipy, and split stability."""
import numpy as np
import pytest

from geo_distill.data import val_split
from geo_distill.losses import cosine_regression_loss, kl_sim_loss, mse_sim_loss
from geo_distill.metrics import similarity_agreement


@pytest.fixture(scope="module")
def emb():
    rng = np.random.default_rng(0)
    return rng.standard_normal((24, 16)).astype(np.float32)


def test_losses_at_identity(emb):
    assert float(cosine_regression_loss(emb, emb)) == pytest.approx(0.0, abs=1e-6)
    assert float(mse_sim_loss(emb, emb)) == pytest.approx(0.0, abs=1e-6)
    # KL of a distribution against itself is 0; the loss is the cross-entropy,
    # so at identity it equals the entropy — finite and non-negative.
    ce = float(kl_sim_loss(emb, emb, 0.05))
    assert np.isfinite(ce) and ce >= 0.0


def test_losses_finite_and_ordered(emb):
    rng = np.random.default_rng(1)
    other = rng.standard_normal(emb.shape).astype(np.float32)
    for fn in (cosine_regression_loss, mse_sim_loss):
        mismatched = float(fn(emb, other))
        assert np.isfinite(mismatched) and mismatched > float(fn(emb, emb))
    assert float(kl_sim_loss(emb, other, 0.05)) > float(kl_sim_loss(emb, emb, 0.05))


def test_similarity_agreement_matches_scipy(emb):
    from scipy import stats

    rng = np.random.default_rng(2)
    other = (emb + 0.5 * rng.standard_normal(emb.shape)).astype(np.float32)
    m = similarity_agreement(emb, other)

    def unit(x):
        x = x.astype(np.float64)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    s, t = unit(emb), unit(other)
    iu = np.triu_indices(len(emb), k=1)
    su, tu = (s @ s.T)[iu], (t @ t.T)[iu]
    assert m["pearson"] == pytest.approx(stats.pearsonr(su, tu).statistic, abs=1e-9)
    assert m["spearman"] == pytest.approx(stats.spearmanr(su, tu).statistic, abs=1e-9)


def test_similarity_agreement_does_not_mutate_input(emb):
    a = emb.astype(np.float64)          # float64: np.asarray would alias this
    b = (emb * 2).astype(np.float64)
    a_before, b_before = a.copy(), b.copy()
    similarity_agreement(a, b)
    np.testing.assert_array_equal(a, a_before)
    np.testing.assert_array_equal(b, b_before)


def test_val_split_stable_under_growth(sentences):
    """A sentence's train/val assignment depends only on its text: embedding
    more data must not move the benchmark."""
    frac, seed = 0.3, 0
    tr_small, va_small = val_split(sentences[:60], frac, seed)
    tr_full, va_full = val_split(sentences, frac, seed)
    assert len(va_small) >= 2 and len(va_full) >= 2  # hash path, not the fallback
    val_small_texts = {sentences[i] for i in va_small}
    val_full_texts = {sentences[i] for i in va_full}
    assert val_small_texts == {s for s in sentences[:60] if s in val_full_texts}


def test_val_split_tiny_corpus_fallback(sentences):
    tiny = sentences[:5]
    tr, va = val_split(tiny, 0.1, seed=0)
    assert len(va) >= 2 and len(tr) >= 1
    assert sorted(list(tr) + list(va)) == list(range(len(tiny)))
