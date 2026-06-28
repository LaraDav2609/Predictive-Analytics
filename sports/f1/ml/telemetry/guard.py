"""Stage-aware leakage guard for telemetry-derived model inputs."""

from __future__ import annotations

from typing import Any

from sports.f1.ml.telemetry.types import TelemetryFeaturePayload
from sports.f1.predictor.data_quality.leakage_guard import leakage_guard_status


def telemetry_leakage_guard_status(
    payload: TelemetryFeaturePayload | dict[str, Any] | None,
    *,
    stage: str,
    session: str | None = None,
    live: bool | None = None,
) -> dict[str, Any]:
    """Return whether telemetry evidence is allowed for the selected replay stage."""

    source = _payload_dict(payload)
    session_key = str(session or source.get("session") or "race").lower()
    data_quality = source.get("data_quality") or {}
    live_flag = bool(live if live is not None else data_quality.get("live"))
    if source.get("ok") is False or not source.get("driver_features") or data_quality.get("estimated"):
        groups = ["historical", "metadata"]
    else:
        groups = _telemetry_feature_groups(session_key, str(source.get("source_mode") or ""), live_flag)
    guard = leakage_guard_status(stage=stage, feature_groups=groups)
    guard.update({
        "telemetry": True,
        "session": session_key,
        "source_mode": source.get("source_mode"),
        "observed_feature_groups": groups,
        "probability_influence_allowed": guard.get("status") == "ok",
    })
    return guard


def _telemetry_feature_groups(session: str, source_mode: str, live: bool) -> list[str]:
    groups = ["metadata"]
    text = f"{session} {source_mode}".lower()
    if any(token in text for token in ("fp1", "fp2", "fp3", "practice")):
        groups.append("practice_pace")
    elif "qual" in text:
        groups.extend(["qualifying", "grid"])
    elif live or "live" in text or "race" in text:
        groups.extend(["live_timing", "race_control"])
    else:
        groups.append("historical")
    return sorted(set(groups))


def _payload_dict(payload: TelemetryFeaturePayload | dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, TelemetryFeaturePayload):
        return payload.model_dump(mode="json")
    return dict(payload)
