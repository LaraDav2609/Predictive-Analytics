"""Tests for f1_ml.core neural-net uncertainty heads — DeepEnsemble + MC dropout.

Skipped when torch isn't available; these are smoke tests that check the
training loop runs and returns sensible (mean, std) shapes.
"""

from __future__ import annotations

import numpy as np
import pytest


# Importorskip at module level so collection itself doesn't blow up if torch is gone.
torch = pytest.importorskip("torch")


import torch.nn as nn  # noqa: E402  (after importorskip)

from f1_ml.core.deep_ensemble import DeepEnsemble  # noqa: E402
from f1_ml.core.mc_dropout import mc_dropout_predict  # noqa: E402


# ----------------------------------------------------------------- helpers

def _make_dataset(n: int = 200, n_features: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, n_features, generator=g)
    w = torch.tensor([0.5, -0.2, 0.1, 0.3])
    y = X @ w + 0.05 * torch.randn(n, generator=g)
    return X, y


def _mlp_factory(in_features: int = 4, out_features: int = 1, dropout: float = 0.0):
    def factory():
        return nn.Sequential(
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_features),
        )
    return factory


# --------------------------------------------------------------- DeepEnsemble

def test_deep_ensemble_fits_k_models():
    X, y = _make_dataset()
    ens = DeepEnsemble(model_factory=_mlp_factory(), k=3, lr=0.01)
    ens.fit(X, y, epochs=5)
    assert len(ens.models) == 3


def test_deep_ensemble_predict_returns_mean_std():
    X, y = _make_dataset(n=100)
    ens = DeepEnsemble(model_factory=_mlp_factory(), k=4, lr=0.01).fit(X, y, epochs=5)
    mean, std = ens.predict(X[:10])
    assert mean.shape == (10, 1) or mean.shape == (10,)
    assert std.shape == mean.shape
    assert (std >= 0).all()


def test_deep_ensemble_std_nonzero_across_independent_inits():
    """Different random seeds → different fits → ensemble std > 0."""
    X, y = _make_dataset(n=80)
    ens = DeepEnsemble(model_factory=_mlp_factory(), k=5, lr=0.01).fit(X, y, epochs=3)
    _, std = ens.predict(X[:20])
    assert std.mean() > 0


def test_deep_ensemble_predict_without_fit_raises():
    ens = DeepEnsemble(model_factory=_mlp_factory(), k=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        ens.predict(torch.randn(3, 4))


def test_deep_ensemble_invalid_k_raises():
    with pytest.raises(ValueError):
        DeepEnsemble(model_factory=_mlp_factory(), k=0)


def test_deep_ensemble_unknown_loss_raises():
    with pytest.raises(ValueError):
        DeepEnsemble(model_factory=_mlp_factory(), loss="huber")


# ----------------------------------------------------------------- MC dropout

def test_mc_dropout_returns_finite_mean_std():
    X, _ = _make_dataset(n=50)
    model = _mlp_factory(dropout=0.3)()
    mean, std = mc_dropout_predict(model, X[:10], n_samples=20)
    assert mean.shape == (10, 1) or mean.shape == (10,)
    assert std.shape == mean.shape
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()


def test_mc_dropout_zero_std_without_dropout_layer():
    """Pure deterministic model → std should be zero."""
    X, _ = _make_dataset(n=20)
    model = _mlp_factory(dropout=0.0)()
    _, std = mc_dropout_predict(model, X[:5], n_samples=10)
    # float32 sample-stddev floor is around 3e-8; allow generous slack.
    assert np.allclose(std, 0.0, atol=1e-6)


def test_mc_dropout_invalid_n_samples_raises():
    model = _mlp_factory()()
    with pytest.raises(ValueError):
        mc_dropout_predict(model, torch.randn(2, 4), n_samples=0)


def test_mc_dropout_apply_sigmoid_keeps_probabilities_in_unit_interval():
    X, _ = _make_dataset(n=30)
    model = _mlp_factory(dropout=0.3)()
    mean, _ = mc_dropout_predict(model, X[:10], n_samples=10, apply_sigmoid=True)
    assert ((mean >= 0) & (mean <= 1)).all()
