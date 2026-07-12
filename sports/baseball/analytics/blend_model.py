"""Trained logistic blend of the MLB model's signals.

The hand-tuned model adds fixed per-signal weights in probability space
(``HOME_FIELD_EDGE`` + ``FORM_WEIGHT`` * form + ``PITCHER_WEIGHT`` * pitcher).
This module learns those weights from data instead: a logistic regression over the
raw signals (``strength``, ``form_gap``, ``pitcher_gap``, ``pitcher_present``) whose
intercept absorbs the home-field edge. It is the baseball analog of the F1 trained
win model (``sports/f1/ml/training/win_model.py`` → ``artifacts/f1_win_model.json``):
fit offline, deployed as a tiny JSON artifact, applied in **pure Python** at serve.

Applying the blend decomposes cleanly in LOG-ODDS space — the final probability is
``sigmoid(w0 + Σ w_i x_i)`` and each ``w_i x_i`` is that signal's log-odds push — so
the dashboard can still show a "why this pick" breakdown, now with learned weights.
Enabled only when it beats the hand-tuned model through the same backtest gate.
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Optional

# Raw signals the blend learns weights over. The original four are always present; the
# run-environment and schedule-fatigue signals are appended so the blend CAN learn them
# when a fit is run with those overlays on. They are 0 / neutral in the feature vector
# when their overlay is absent, and any deployed artifact whose ``coef`` predates them
# simply gets a 0 weight for the new keys (``apply_blend`` defaults missing coefs to 0),
# so extending this list is backward-compatible with the currently enabled artifact.
FEATURE_ORDER = [
    "strength", "form_gap", "pitcher_gap", "pitcher_present",
    "run_env_index", "wind_out_mph", "env_present",
    "rest_gap", "density_gap", "fatigue_present",
    "bullpen_gap", "bullpen_present",
    "lineup_gap", "lineup_present",
]

_ARTIFACT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "artifacts", "baseball_blend.json")
)

_lock = threading.Lock()
_cache: Optional[dict] = None
_cache_mtime: float = -1.0


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def apply_blend(features: dict, weights: dict) -> dict:
    """``sigmoid(intercept + Σ w_i x_i)`` with per-feature log-odds attribution.

    Returns ``{value, intercept, logit, contributions}`` where ``contributions[f]``
    is ``w_f * x_f`` (the signal's push on the log-odds of a home win)."""
    coef = weights.get("coef", {})
    intercept = float(weights.get("intercept", 0.0))
    contributions = {f: round(float(coef.get(f, 0.0)) * float(features.get(f, 0.0)), 6)
                     for f in FEATURE_ORDER}
    logit = intercept + sum(contributions.values())
    return {"value": round(_sigmoid(logit), 4), "intercept": round(intercept, 6),
            "logit": round(logit, 6), "contributions": contributions}


def _load() -> Optional[dict]:
    global _cache, _cache_mtime
    with _lock:
        try:
            mtime = os.path.getmtime(_ARTIFACT_PATH)
        except OSError:
            _cache, _cache_mtime = None, -1.0
            return None
        if _cache is not None and mtime == _cache_mtime:
            return _cache
        try:
            with open(_ARTIFACT_PATH, "r", encoding="utf-8") as fh:
                _cache = json.load(fh)
            _cache_mtime = mtime
        except (OSError, json.JSONDecodeError):
            _cache = None
        return _cache


def state() -> dict:
    art = _load()
    if not art:
        return {"fitted": False, "enabled": False, "path": _ARTIFACT_PATH}
    return {**art, "fitted": True, "path": _ARTIFACT_PATH}


def blend(features: dict) -> dict:
    """Apply the enabled blend to a feature vector. Returns ``{applied, value, ...}``;
    when no enabled blend exists ``applied`` is False and callers keep the
    hand-tuned probability."""
    art = _load()
    if not art or not art.get("enabled") or "weights" not in art:
        return {"applied": False}
    out = apply_blend(features, art["weights"])
    return {"applied": True, "model_version": art.get("model_version", "mlb-blend-v1"), **out}


def save(artifact: dict) -> dict:
    global _cache, _cache_mtime
    os.makedirs(os.path.dirname(_ARTIFACT_PATH), exist_ok=True)
    with _lock:
        with open(_ARTIFACT_PATH, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)
        _cache, _cache_mtime = None, -1.0
    return state()


def set_enabled(on: bool) -> dict:
    art = _load()
    if not art:
        return {"fitted": False, "enabled": False, "path": _ARTIFACT_PATH,
                "reason": "no blend fitted yet"}
    return save({**art, "enabled": bool(on)})
