"""Tests for serve-time Platt calibration (pure-python apply + artifact I/O)."""
import math

import pytest

from sports.baseball.analytics import serve_calibration as sc


def test_apply_platt_identity():
    # a=1, b=0 is the identity map (within float tolerance).
    for p in (0.1, 0.4, 0.5, 0.73, 0.9):
        assert abs(sc.apply_platt(p, 1.0, 0.0) - p) < 1e-9


def test_apply_platt_is_monotonic_and_bounded():
    a, b = 1.4, -0.2
    vals = [sc.apply_platt(p, a, b) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert all(0.0 < v < 1.0 for v in vals)
    assert vals == sorted(vals)                       # monotone increasing in p


def test_apply_platt_handles_extremes():
    assert 0.0 < sc.apply_platt(0.0, 1.0, 0.0) < 1e-6
    assert 1.0 - 1e-6 < sc.apply_platt(1.0, 1.0, 0.0) < 1.0


@pytest.fixture
def temp_artifact(tmp_path, monkeypatch):
    path = str(tmp_path / "baseball_calibrator.json")
    monkeypatch.setattr(sc, "_ARTIFACT_PATH", path)
    monkeypatch.setattr(sc, "_cache", None)
    monkeypatch.setattr(sc, "_cache_mtime", -1.0)
    return path


def test_calibrate_noop_without_artifact(temp_artifact):
    out = sc.calibrate(0.62)
    assert out["applied"] is False and out["value"] == 0.62 and out["delta"] == 0.0


def test_save_load_and_enable_roundtrip(temp_artifact):
    sc.save({"method": "platt", "params": {"a": 1.5, "b": -0.25}, "enabled": False})
    # Disabled → served value is the raw value.
    assert sc.calibrate(0.62)["applied"] is False
    # Enable → served value is the Platt map, and state reflects it.
    sc.set_enabled(True)
    out = sc.calibrate(0.62)
    assert out["applied"] is True
    assert out["method"] == "platt"
    assert abs(out["value"] - sc.apply_platt(0.62, 1.5, -0.25)) < 5e-5   # value is rounded 4dp
    assert sc.state()["enabled"] is True


def test_enable_without_fit_is_graceful(temp_artifact):
    st = sc.set_enabled(True)
    assert st["fitted"] is False and st["enabled"] is False
