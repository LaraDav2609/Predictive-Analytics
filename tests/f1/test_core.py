"""Tests for sports.f1.ml.core — GBM heads + conformal calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports.f1.ml.core.conformal import ConformalCalibrator
from sports.f1.ml.core.gbm_dnf import GBMDNFModel
from sports.f1.ml.core.gbm_overtake import GBMOvertakeModel
from sports.f1.ml.core.gbm_pace import GBMPaceModel


# -------------------------------------------------------------------- gbm_pace

def _make_pace_dataset(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic dataset where lap_time ≈ 80 + 0.05 * tire_age + 0.2 * dirty_air."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "tire_age": rng.integers(0, 30, size=n),
        "dirty_air": rng.uniform(0, 1.5, size=n),
        "fuel_kg": rng.uniform(20, 110, size=n),
    })
    y = (80.0 + 0.05 * X["tire_age"] + 0.2 * X["dirty_air"]
         + 0.03 * X["fuel_kg"] + rng.normal(0, 0.1, size=n)).to_numpy()
    return X, y


def test_gbm_pace_recovers_synthetic_relationship():
    X, y = _make_pace_dataset(n=600)
    train_X, test_X = X.iloc[:500], X.iloc[500:]
    train_y, test_y = y[:500], y[500:]

    model = GBMPaceModel(n_estimators=200, learning_rate=0.05).fit(train_X, train_y)
    preds = model.predict(test_X)

    # MAE should be small relative to the noise floor (σ=0.1).
    mae = np.mean(np.abs(preds - test_y))
    assert mae < 0.3, f"MAE {mae:.3f} too high"


def test_gbm_pace_predict_without_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GBMPaceModel().predict(pd.DataFrame({"x": [1.0]}))


def test_gbm_pace_quantile_predictions_are_ordered():
    X, y = _make_pace_dataset(n=400)
    model = GBMPaceModel(n_estimators=100).fit_quantiles(
        X, y, alphas=(0.1, 0.5, 0.9)
    )
    qs = model.predict_with_quantiles(X.iloc[:50], alphas=(0.1, 0.5, 0.9))
    assert qs.shape == (50, 3)
    # On most rows: q10 ≤ q50 ≤ q90 (monotone).
    monotone = ((qs[:, 0] <= qs[:, 1]) & (qs[:, 1] <= qs[:, 2])).mean()
    assert monotone >= 0.9


def test_gbm_pace_quantile_without_fit_raises():
    with pytest.raises(RuntimeError, match="fit_quantiles"):
        GBMPaceModel().predict_with_quantiles(pd.DataFrame({"x": [1.0]}))


# -------------------------------------------------------------------- gbm_dnf

def _make_dnf_dataset(n: int = 800, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic: P(DNF) = sigmoid(-3 + 0.04 * lap + 0.5 * mech_concern)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "lap": rng.integers(1, 60, size=n),
        "mech_concern": rng.uniform(0, 2, size=n),
        "engine_age": rng.integers(0, 8, size=n),
    })
    logits = -3.0 + 0.04 * X["lap"] + 0.5 * X["mech_concern"]
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(0, 1, size=n) < p).astype(int).to_numpy()
    return X, y


def test_gbm_dnf_hazard_increases_with_lap_and_concern():
    X, y = _make_dnf_dataset(n=1500)
    model = GBMDNFModel(n_estimators=200).fit(X, y)

    # Construct probe rows: low-risk vs high-risk.
    low = pd.DataFrame({"lap": [5], "mech_concern": [0.1], "engine_age": [1]})
    high = pd.DataFrame({"lap": [55], "mech_concern": [1.8], "engine_age": [7]})
    h_low = model.hazard_per_lap(low)[0]
    h_high = model.hazard_per_lap(high)[0]
    assert 0 <= h_low <= 1 and 0 <= h_high <= 1
    assert h_high > h_low, f"high-risk hazard {h_high} should exceed low-risk {h_low}"


def test_gbm_dnf_hazard_in_unit_interval():
    X, y = _make_dnf_dataset(n=400)
    model = GBMDNFModel(n_estimators=100).fit(X, y)
    hazards = model.hazard_per_lap(X)
    assert (hazards >= 0).all() and (hazards <= 1).all()


def test_gbm_dnf_predict_without_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GBMDNFModel().hazard_per_lap(pd.DataFrame({"lap": [1]}))


# --------------------------------------------------------------- gbm_overtake

def _make_overtake_dataset(n: int = 600, seed: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic: pass succeeds when (pace_delta + DRS bonus - gap) > 0."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "gap_ahead_s": rng.uniform(0.1, 2.0, size=n),
        "pace_delta_s": rng.uniform(-0.5, 1.5, size=n),
        "drs_available": rng.integers(0, 2, size=n),
    })
    score = X["pace_delta_s"] + 0.3 * X["drs_available"] - 0.5 * X["gap_ahead_s"]
    y = (rng.uniform(0, 1, size=n) < 1.0 / (1.0 + np.exp(-score * 3))).astype(int).to_numpy()
    return X, y


def test_gbm_overtake_pass_more_likely_with_pace_advantage():
    X, y = _make_overtake_dataset(n=800)
    model = GBMOvertakeModel(n_estimators=200).fit(X, y)

    fast = pd.DataFrame({"gap_ahead_s": [0.3], "pace_delta_s": [1.2], "drs_available": [1]})
    slow = pd.DataFrame({"gap_ahead_s": [0.3], "pace_delta_s": [-0.4], "drs_available": [0]})
    assert model.predict_proba(fast)[0] > model.predict_proba(slow)[0]


def test_gbm_overtake_predict_without_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        GBMOvertakeModel().predict_proba(pd.DataFrame({"x": [1.0]}))


# ----------------------------------------------------------------- conformal

def test_conformal_threshold_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 500
    probs = rng.uniform(0, 1, size=n)
    outcomes = (rng.uniform(0, 1, size=n) < probs).astype(int)
    cal = ConformalCalibrator.fit(probs, outcomes, alpha=0.1)
    assert 0.0 <= cal.threshold <= 1.0


def test_conformal_binary_prediction_set_well_formed():
    rng = np.random.default_rng(0)
    n = 1000
    probs = rng.uniform(0.05, 0.95, size=n)
    outcomes = (rng.uniform(0, 1, size=n) < probs).astype(int)
    cal = ConformalCalibrator.fit(probs, outcomes, alpha=0.1)

    # Confident-1 prob should yield {1}.
    assert cal.binary_prediction_set(0.99) == [1]
    # Confident-0 prob should yield {0}.
    assert cal.binary_prediction_set(0.01) == [0]
    # Mid prob may yield {0, 1} (both/uncertain).
    mid_set = cal.binary_prediction_set(0.5)
    assert set(mid_set).issubset({0, 1})


def test_conformal_empirical_coverage_at_least_1_minus_alpha():
    rng = np.random.default_rng(42)
    n = 3000
    probs = rng.uniform(0.05, 0.95, size=n)
    outcomes = (rng.uniform(0, 1, size=n) < probs).astype(int)
    # Split: 60% calibration / 40% test.
    cut = int(0.6 * n)
    cal = ConformalCalibrator.fit(probs[:cut], outcomes[:cut], alpha=0.1)
    coverage = cal.empirical_coverage(probs[cut:], outcomes[cut:])
    # Marginal coverage should be at least 1 - α (with some Monte Carlo noise).
    assert coverage >= 0.85, f"coverage {coverage:.3f} below 1-α=0.9 minus tolerance"


def test_conformal_multiclass_prediction_set_includes_top_class():
    """For a confident multi-class prediction, the top class should always
    be in the conformal prediction set."""
    rng = np.random.default_rng(0)
    n_classes = 5
    n = 500
    # Generate per-row probabilities with peak at the true class.
    outcomes = rng.integers(0, n_classes, size=n)
    probs = np.full((n, n_classes), 0.05)
    probs[np.arange(n), outcomes] = 0.8
    probs = probs / probs.sum(axis=1, keepdims=True)
    cal = ConformalCalibrator.fit(probs, outcomes, alpha=0.1)

    # Confident peak at class 2.
    confident = np.array([0.02, 0.02, 0.92, 0.02, 0.02])
    assert 2 in cal.prediction_set(confident)


def test_conformal_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ConformalCalibrator.fit(np.array([0.5]), np.array([1]), alpha=0.0)
    with pytest.raises(ValueError):
        ConformalCalibrator.fit(np.array([0.5]), np.array([1]), alpha=1.0)


def test_conformal_rejects_empty_calibration_set():
    with pytest.raises(ValueError):
        ConformalCalibrator.fit(np.array([]), np.array([]))


def test_conformal_transform_is_identity():
    """Conformal doesn't transform probabilities directly; it produces sets."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, size=20)
    outcomes = rng.integers(0, 2, size=20)
    cal = ConformalCalibrator.fit(probs, outcomes, alpha=0.1)
    np.testing.assert_array_equal(cal.transform(probs), probs)
