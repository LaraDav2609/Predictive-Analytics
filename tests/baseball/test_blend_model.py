"""Tests for the trained logistic blend (pure-python apply + artifact I/O)."""
import math

import pytest

from sports.baseball.analytics import blend_model as bm


def _w(intercept=0.0, strength=0.0, form_gap=0.0, pitcher_gap=0.0, pitcher_present=0.0):
    return {"intercept": intercept,
            "coef": {"strength": strength, "form_gap": form_gap,
                     "pitcher_gap": pitcher_gap, "pitcher_present": pitcher_present}}


def test_apply_blend_intercept_only():
    # Only the intercept fires → sigmoid(intercept), no signal contributions.
    out = bm.apply_blend({}, _w(intercept=0.5))
    assert abs(out["value"] - 1.0 / (1.0 + math.exp(-0.5))) < 1e-4
    assert all(v == 0.0 for v in out["contributions"].values())


def test_apply_blend_contributions_are_weight_times_feature():
    feats = {"strength": 0.2, "form_gap": -0.1, "pitcher_gap": 1.5, "pitcher_present": 1.0}
    w = _w(intercept=0.1, strength=1.4, form_gap=0.2, pitcher_gap=0.16, pitcher_present=-0.4)
    out = bm.apply_blend(feats, w)
    assert abs(out["contributions"]["strength"] - 1.4 * 0.2) < 1e-6
    assert abs(out["contributions"]["pitcher_gap"] - 0.16 * 1.5) < 1e-6
    # logit is intercept + sum of contributions
    assert abs(out["logit"] - (0.1 + sum(out["contributions"].values()))) < 1e-6


def test_apply_blend_stronger_home_strength_raises_prob():
    w = _w(intercept=0.0, strength=1.4)
    lo = bm.apply_blend({"strength": -0.2}, w)["value"]
    hi = bm.apply_blend({"strength": 0.2}, w)["value"]
    assert hi > 0.5 > lo


@pytest.fixture
def temp_artifact(tmp_path, monkeypatch):
    path = str(tmp_path / "baseball_blend.json")
    monkeypatch.setattr(bm, "_ARTIFACT_PATH", path)
    monkeypatch.setattr(bm, "_cache", None)
    monkeypatch.setattr(bm, "_cache_mtime", -1.0)
    return path


def test_blend_noop_without_artifact(temp_artifact):
    assert bm.blend({"strength": 0.2})["applied"] is False


def test_save_enable_roundtrip(temp_artifact):
    bm.save({"model_version": "mlb-blend-v1", "weights": _w(intercept=0.1, strength=1.4),
             "enabled": False})
    assert bm.blend({"strength": 0.2})["applied"] is False      # disabled → no-op
    bm.set_enabled(True)
    out = bm.blend({"strength": 0.2})
    assert out["applied"] is True and out["model_version"] == "mlb-blend-v1"
    assert abs(out["value"] - bm.apply_blend({"strength": 0.2}, _w(intercept=0.1, strength=1.4))["value"]) < 1e-9
    assert bm.state()["enabled"] is True


def test_enable_without_fit_is_graceful(temp_artifact):
    st = bm.set_enabled(True)
    assert st["fitted"] is False and st["enabled"] is False
