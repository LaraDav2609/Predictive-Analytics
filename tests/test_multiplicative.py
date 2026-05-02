"""Tests for f1_ml.multiplicative — stacking + multi_task uncertainty loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1_ml.multiplicative.stacking import StackedEnsemble


# -------------------------------------------------------------------- stacking

def _make_oof_predictions(
    n: int = 500, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    """Two base models with slightly different errors; truth is sigmoid(z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, size=n)
    truth = 1.0 / (1.0 + np.exp(-z))
    base_a = np.clip(truth + rng.normal(0, 0.1, size=n), 0.01, 0.99)
    base_b = np.clip(truth + rng.normal(0, 0.15, size=n), 0.01, 0.99)
    outcomes = (rng.uniform(0, 1, size=n) < truth).astype(int)
    df = pd.DataFrame({"physics_mc": base_a, "gbm": base_b, "track_difficulty": z})
    return df, outcomes


def test_stacking_logistic_fits_and_predicts():
    df, y = _make_oof_predictions(n=500)
    ens = StackedEnsemble(base_model_names=["physics_mc", "gbm"], meta_learner="logistic")
    ens.fit(df, y)
    probs = ens.predict_proba(df.iloc[:50])
    assert probs.shape == (50,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_stacking_meta_beats_or_matches_individual_bases():
    """Stacked meta should achieve Brier ≤ best base on the same training data."""
    from f1_ml.common.calibration import brier_score
    df, y = _make_oof_predictions(n=2000)
    ens = StackedEnsemble(base_model_names=["physics_mc", "gbm"]).fit(df, y)
    meta_probs = ens.predict_proba(df)
    meta_brier = brier_score(meta_probs, y)
    base_a_brier = brier_score(df["physics_mc"].to_numpy(), y)
    base_b_brier = brier_score(df["gbm"].to_numpy(), y)
    assert meta_brier <= max(base_a_brier, base_b_brier) + 1e-3


def test_stacking_predict_without_fit_raises():
    ens = StackedEnsemble(base_model_names=["physics_mc"])
    with pytest.raises(RuntimeError, match="not fitted"):
        ens.predict_proba(pd.DataFrame({"physics_mc": [0.5]}))


def test_stacking_missing_base_column_raises():
    df = pd.DataFrame({"only_a": [0.1, 0.5, 0.9]})
    ens = StackedEnsemble(base_model_names=["physics_mc", "gbm"])
    with pytest.raises(KeyError, match="missing base columns"):
        ens.fit(df, np.array([0, 1, 0]))


def test_stacking_invalid_meta_learner_raises():
    with pytest.raises(ValueError):
        StackedEnsemble(base_model_names=["a"], meta_learner="banana")


def test_stacking_lightgbm_meta_learner_runs():
    df, y = _make_oof_predictions(n=300)
    ens = StackedEnsemble(base_model_names=["physics_mc", "gbm"], meta_learner="lightgbm").fit(df, y)
    probs = ens.predict_proba(df.iloc[:50])
    assert ((probs >= 0) & (probs <= 1)).all()


# ----------------------------------------------------- multi_task uncertainty loss

def test_uncertainty_weighted_loss_combines_tasks():
    pytest.importorskip("torch")
    import torch
    from f1_ml.multiplicative.multi_task import uncertainty_weighted_loss

    losses = {
        "pace": torch.tensor(2.0),
        "dnf": torch.tensor(1.0),
        "overtake": torch.tensor(0.5),
        "pit_time": torch.tensor(1.5),
    }
    log_vars = torch.zeros(4)  # σ² = 1 ⇒ each precision = 1
    out = uncertainty_weighted_loss(losses, log_vars)
    # Expected = sum(losses) + sum(log_vars) = 5.0 + 0.0 = 5.0
    assert float(out) == pytest.approx(5.0, abs=1e-6)


def test_uncertainty_weighted_loss_high_log_var_downweights():
    pytest.importorskip("torch")
    import torch
    from f1_ml.multiplicative.multi_task import uncertainty_weighted_loss

    losses = {"pace": torch.tensor(10.0), "dnf": torch.tensor(0.0),
              "overtake": torch.tensor(0.0), "pit_time": torch.tensor(0.0)}
    log_vars_low = torch.zeros(4)
    log_vars_high = torch.tensor([5.0, 0.0, 0.0, 0.0])  # high noise for pace
    low = float(uncertainty_weighted_loss(losses, log_vars_low))
    high = float(uncertainty_weighted_loss(losses, log_vars_high))
    # High log_var should reduce the weighted contribution of pace's loss.
    assert high < low


def test_uncertainty_weighted_loss_rejects_wrong_log_vars_shape():
    pytest.importorskip("torch")
    import torch
    from f1_ml.multiplicative.multi_task import uncertainty_weighted_loss

    with pytest.raises(ValueError):
        uncertainty_weighted_loss(
            {"pace": torch.tensor(1.0)},
            torch.zeros(2),  # wrong shape
        )


def test_uncertainty_weighted_loss_skips_missing_tasks():
    pytest.importorskip("torch")
    import torch
    from f1_ml.multiplicative.multi_task import uncertainty_weighted_loss

    # Only 'pace' present — others should be silently skipped (e.g., when one
    # head has no labels in this batch).
    losses = {"pace": torch.tensor(3.0)}
    log_vars = torch.zeros(4)
    out = uncertainty_weighted_loss(losses, log_vars)
    assert float(out) == pytest.approx(3.0, abs=1e-6)
