"""Rollout policy helpers for the telemetry simulator."""

from __future__ import annotations

import os
from typing import Any


def telemetry_rollout_policy(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve telemetry rollout flags from explicit overrides and environment."""

    policy = dict(overrides or {})
    enabled = policy.get("enabled")
    warn_only = policy.get("warn_only")
    min_confidence = policy.get("min_confidence")
    if enabled is None:
        enabled = _env_bool("F1_TELEMETRY_MODEL_ENABLED", True)
    if warn_only is None:
        warn_only = _env_bool("F1_TELEMETRY_MODEL_WARN_ONLY", False)
    try:
        min_confidence_value = float(
            min_confidence
            if min_confidence is not None
            else os.getenv("F1_TELEMETRY_MIN_CONFIDENCE", "0.20")
        )
    except (TypeError, ValueError):
        min_confidence_value = 0.20
    return {
        "enabled": bool(enabled),
        "warn_only": bool(warn_only),
        "min_confidence": max(0.0, min(1.0, min_confidence_value)),
        "source": str(policy.get("source") or os.getenv("F1_TELEMETRY_SOURCE") or "auto"),
    }


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}
