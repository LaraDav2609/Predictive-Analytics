"""Serve-time probability calibration for the MLB model.

The backtest fits a Platt scaler and proves it lowers Brier/log-loss out-of-sample,
but that only helps traders if it's actually applied to the numbers the dashboard
serves. This module persists the fitted scaler as a tiny JSON artifact and applies
it at serve time in **pure Python** — no numpy / sklearn needed on the request path,
mirroring the F1 serve-artifact pattern (``artifacts/f1_win_model.json``).

A Platt scaler is just two floats: ``calibrated = sigmoid(a * logit(p) + b)``.
``a=1, b=0`` is the identity. The artifact also carries provenance (season it was
fit on, sample size, raw-vs-calibrated Brier) and an ``enabled`` flag so calibration
can be fit-then-reviewed before it changes served probabilities.
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Optional

_ARTIFACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "artifacts", "baseball_calibrator.json"
)
_ARTIFACT_PATH = os.path.abspath(_ARTIFACT_PATH)

_lock = threading.Lock()
_cache: Optional[dict] = None
_cache_mtime: float = -1.0

_EPS = 1e-9


def apply_platt(p: float, a: float, b: float) -> float:
    """calibrated = sigmoid(a * logit(p) + b). Pure Python; clamps p away from 0/1."""
    p = min(1.0 - _EPS, max(_EPS, float(p)))
    logit = math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-(a * logit + b)))


def _load() -> Optional[dict]:
    """Read the artifact, cached and invalidated on file mtime."""
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
    """Current calibrator state for the pipeline monitor / API."""
    art = _load()
    if not art:
        return {"fitted": False, "enabled": False, "path": _ARTIFACT_PATH}
    return {**art, "fitted": True, "path": _ARTIFACT_PATH}


def calibrate(p: float) -> dict:
    """Apply the enabled calibrator to a home win probability.

    Returns ``{value, raw, applied, method, delta}``. When no enabled calibrator is
    present ``value == raw`` and ``applied`` is False, so callers can always trust
    ``value`` as the number to serve.
    """
    art = _load()
    raw = float(p)
    if not art or not art.get("enabled") or "params" not in art:
        return {"value": round(raw, 4), "raw": round(raw, 4), "applied": False,
                "method": None, "delta": 0.0}
    a = float(art["params"]["a"]); b = float(art["params"]["b"])
    val = apply_platt(raw, a, b)
    return {"value": round(val, 4), "raw": round(raw, 4), "applied": True,
            "method": art.get("method", "platt"), "delta": round(val - raw, 4)}


def save(artifact: dict) -> dict:
    """Persist the calibrator artifact (invalidating the cache) and return state()."""
    global _cache, _cache_mtime
    os.makedirs(os.path.dirname(_ARTIFACT_PATH), exist_ok=True)
    with _lock:
        with open(_ARTIFACT_PATH, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)
        _cache, _cache_mtime = None, -1.0
    return state()


def set_enabled(on: bool) -> dict:
    """Flip the enabled flag on the existing artifact (no refit)."""
    art = _load()
    if not art:
        return {"fitted": False, "enabled": False, "path": _ARTIFACT_PATH,
                "reason": "no calibrator fitted yet"}
    art = {**art, "enabled": bool(on)}
    return save(art)
