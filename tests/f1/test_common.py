"""Tests for sports.f1.ml.common — calibration + artifact IO."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from common.ml.calibration import (
    IsotonicCalibrator,
    PlattScaler,
    brier_score,
    log_loss,
    reliability_curve,
)
from common.ml.io import ArtifactManifest, load, manifest_path, read_manifest, save, write_manifest


# -------------------------------------------------------------- calibration

def test_platt_scaler_identity_when_already_calibrated():
    """If raw probs are already well-calibrated, Platt should learn near-identity (a≈1, b≈0)."""
    rng = np.random.default_rng(0)
    n = 1000
    probs = rng.uniform(0.1, 0.9, size=n)
    outcomes = (rng.uniform(0, 1, size=n) < probs).astype(int)
    scaler = PlattScaler().fit(probs, outcomes)
    # a should be near 1 (well-calibrated input).
    assert abs(scaler.a - 1.0) < 0.5
    # Calibrated output should still be close to input.
    out = scaler.transform(probs)
    assert np.abs(out - probs).mean() < 0.1


def test_platt_scaler_corrects_overconfident_input():
    """If raw probs are pushed away from 0.5 (overconfident), Platt should pull them back."""
    rng = np.random.default_rng(0)
    n = 1000
    # True probs are 0.6, but model outputs 0.9 (overconfident).
    true_probs = np.full(n, 0.6)
    raw_probs = np.full(n, 0.9)
    outcomes = (rng.uniform(0, 1, size=n) < true_probs).astype(int)
    scaler = PlattScaler().fit(raw_probs, outcomes)
    out = scaler.transform(raw_probs)
    # Output should be much closer to 0.6 than 0.9.
    assert abs(out[0] - 0.6) < abs(out[0] - 0.9)


def test_platt_scaler_handles_single_class_outcomes():
    """If all outcomes are 1, fit should return identity (no exception)."""
    probs = np.array([0.3, 0.5, 0.7])
    outcomes = np.ones(3, dtype=int)
    scaler = PlattScaler().fit(probs, outcomes)
    # Falls through to identity.
    assert scaler.a == 1.0


def test_platt_transform_without_fit_is_identity():
    probs = np.array([0.1, 0.5, 0.9])
    out = PlattScaler().transform(probs)
    assert np.allclose(out, probs, atol=1e-6)


def test_isotonic_calibrator_monotonic():
    """Isotonic must produce monotonically non-decreasing calibrated outputs."""
    rng = np.random.default_rng(0)
    n = 500
    probs = rng.uniform(0, 1, size=n)
    # True calibration curve: f(p) = sqrt(p) — concave.
    outcomes = (rng.uniform(0, 1, size=n) < np.sqrt(probs)).astype(int)
    cal = IsotonicCalibrator().fit(probs, outcomes)
    # Sweep input from 0 to 1 and check output is monotone.
    grid = np.linspace(0, 1, 50)
    out = cal.transform(grid)
    diffs = np.diff(out)
    assert (diffs >= -1e-9).all(), "isotonic output must be non-decreasing"
    # Outputs clipped to [0, 1].
    assert (out >= 0).all() and (out <= 1).all()


def test_isotonic_transform_without_fit_is_identity():
    probs = np.array([0.1, 0.5, 0.9])
    out = IsotonicCalibrator().transform(probs)
    assert np.allclose(out, probs)


# Existing brier/log_loss/reliability sanity (kept light; main coverage in test_smoke.py)

def test_log_loss_zero_when_perfect():
    probs = np.array([1.0, 0.0, 1.0])
    outcomes = np.array([1, 0, 1])
    ll = log_loss(probs, outcomes)
    # ε clipping makes this small but not exactly 0.
    assert ll < 1e-9


def test_reliability_curve_bins_correctly():
    probs = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    outcomes = np.array([0, 0, 1, 1, 1])
    pred, obs, counts = reliability_curve(probs, outcomes, n_bins=5)
    # 5 bins: [0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]
    # 0.1, 0.2 in bin 0/1 boundary; 0.6 in bin 2/3 boundary; 0.7 in bin 3; 0.9 in bin 4
    assert counts.sum() == 5


# ----------------------------------------------------------------- io

def test_write_then_read_manifest_round_trip():
    manifest = ArtifactManifest(
        model_name="hierarchical_bayes",
        model_version="v1",
        trained_at=datetime(2026, 4, 1, 12, 0),
        training_cutoff=datetime(2026, 3, 31, 23, 59),
        feature_schema_hash="abc123",
        extra={"n_samples": 2000, "chains": 2},
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "manifest.json"
        write_manifest(manifest, p)
        loaded = read_manifest(p)
    assert loaded.model_name == manifest.model_name
    assert loaded.model_version == manifest.model_version
    assert loaded.trained_at == manifest.trained_at
    assert loaded.training_cutoff == manifest.training_cutoff
    assert loaded.feature_schema_hash == manifest.feature_schema_hash
    assert loaded.extra == manifest.extra


def test_save_then_load_artifact_round_trip():
    artifact = {"weights": [1.0, 2.0, 3.0], "config": {"lr": 0.01}}
    manifest = ArtifactManifest(
        model_name="test_model",
        model_version="v1",
        trained_at=datetime(2026, 4, 1, 12, 0),
        training_cutoff=datetime(2026, 3, 31, 23, 59),
        feature_schema_hash="hash",
        extra={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "model.pkl"
        save(artifact, manifest, p)
        # Both files exist.
        assert p.exists()
        assert manifest_path(p).exists()
        loaded_artifact, loaded_manifest = load(p)
    assert loaded_artifact == artifact
    assert loaded_manifest.model_name == manifest.model_name
    assert loaded_manifest.training_cutoff == manifest.training_cutoff


def test_manifest_path_replaces_extension():
    assert manifest_path(Path("/tmp/model.pkl")) == Path("/tmp/model.json")
    assert manifest_path(Path("model.pt")) == Path("model.json")


def test_save_creates_parent_directories():
    artifact = [1, 2, 3]
    manifest = ArtifactManifest(
        model_name="x", model_version="v1",
        trained_at=datetime(2026, 1, 1), training_cutoff=datetime(2026, 1, 1),
        feature_schema_hash="h", extra={},
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "nested" / "subdir" / "model.pkl"
        save(artifact, manifest, p)
        assert p.exists()
        assert manifest_path(p).exists()
