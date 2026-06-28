"""Artifact manifest support for future learned telemetry heads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sports.f1.ml.telemetry.types import TelemetryFeatureVector


EXPECTED_HEADS = ("pace_delta", "pace_quantile", "dnf_hazard", "overtake", "pit_value")
DEFAULT_ARTIFACT_ENV = "F1_TELEMETRY_ARTIFACT_PATH"


def load_telemetry_artifact_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load lightweight telemetry artifact metadata without importing model runtimes."""

    configured = path or os.getenv(DEFAULT_ARTIFACT_ENV)
    if not configured:
        return _fallback("artifact_not_configured")
    root = Path(configured)
    manifest_path = root / "metadata.json" if root.is_dir() else root
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fallback("artifact_not_found", path=str(manifest_path))
    except json.JSONDecodeError:
        return _fallback("artifact_invalid_json", path=str(manifest_path))
    except OSError as exc:
        return _fallback(f"artifact_read_failed:{exc.__class__.__name__}", path=str(manifest_path))

    heads = raw.get("heads") or raw.get("model_heads") or {}
    present = sorted(str(name) for name, meta in heads.items() if meta is not None)
    missing = [name for name in EXPECTED_HEADS if name not in present]
    return {
        "ok": not missing,
        "configured": True,
        "path": str(manifest_path),
        "artifact_id": raw.get("artifact_id") or raw.get("id"),
        "artifact_version": raw.get("artifact_version") or raw.get("version"),
        "schema_version": raw.get("schema_version") or raw.get("telemetry_schema_version"),
        "trained_heads": present,
        "missing_heads": missing,
        "calibration": raw.get("calibration") or {},
        "feature_schema": raw.get("feature_schema") or raw.get("features") or {},
        "head_definitions": _compact_heads(heads),
        "fallback_reason": None if not missing else "missing_telemetry_heads",
        "raw_metadata": _compact_metadata(raw),
    }


def score_telemetry_artifact_heads(
    artifact_status: dict[str, Any],
    feature: TelemetryFeatureVector,
) -> dict[str, float]:
    """Score JSON-linear learned heads for a feature vector.

    This intentionally supports a small, portable artifact shape before a full
    model runtime is introduced:

    {
      "heads": {
        "pace_delta": {
          "intercept": 0.0,
          "coefficients": {"clean_air_pace_delta_s": 0.6},
          "clamp": [-1.2, 1.2]
        }
      }
    }
    """

    if not artifact_status.get("ok"):
        return {}
    heads = artifact_status.get("head_definitions") or {}
    predictions: dict[str, float] = {}
    for name in EXPECTED_HEADS:
        head = heads.get(name)
        if not isinstance(head, dict):
            continue
        coefficients = head.get("coefficients") or head.get("weights") or {}
        if not isinstance(coefficients, dict):
            continue
        value = _safe_float(head.get("intercept") or head.get("bias") or 0.0) or 0.0
        for feature_name, weight in coefficients.items():
            value += (_feature_value(feature, str(feature_name)) or 0.0) * (_safe_float(weight) or 0.0)
        clamp = head.get("clamp") or head.get("bounds")
        if isinstance(clamp, (list, tuple)) and len(clamp) >= 2:
            lo = _safe_float(clamp[0])
            hi = _safe_float(clamp[1])
            if lo is not None and hi is not None and lo <= hi:
                value = max(lo, min(hi, value))
        predictions[name] = round(float(value), 5)
    return predictions


def _fallback(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "ok": False,
        "configured": bool(extra.get("path")),
        "artifact_id": None,
        "artifact_version": None,
        "trained_heads": [],
        "missing_heads": list(EXPECTED_HEADS),
        "fallback_reason": reason,
        **extra,
    }


def _compact_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        key: raw.get(key)
        for key in ("artifact_id", "artifact_version", "version", "schema_version", "created_at", "training_window", "metrics")
        if key in raw
    }


def _compact_heads(heads: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for name in EXPECTED_HEADS:
        head = heads.get(name)
        if not isinstance(head, dict):
            continue
        compact[name] = {
            key: head.get(key)
            for key in ("intercept", "bias", "coefficients", "weights", "clamp", "bounds", "calibration")
            if key in head
        }
    return compact


def _feature_value(feature: TelemetryFeatureVector, name: str) -> float | None:
    raw = feature.model_dump(mode="python")
    value = raw.get(name)
    if value is None and name.startswith("raw."):
        cursor: Any = raw.get("raw") or {}
        for part in name.split(".")[1:]:
            if not isinstance(cursor, dict):
                return None
            cursor = cursor.get(part)
        value = cursor
    return _safe_float(value)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
