"""Explicit freshness / staleness alerts for the pipeline monitor.

Rolls up the data signals the pipeline already tracks — official results, weather,
sentiment, the live source mode/fallback, and the Redis/ClickHouse bridges — into a
list of structured alerts ``{source, status, severity, message}`` for
``/api/f1/pipeline/health`` and the ``f1:ops:*`` feed.

This is a roll-up of EXISTING signals; it adds no new instrumentation. Per-stream
age-vs-limit checks (e.g. OpenF1 rows older than their ``freshness_seconds`` limit)
are a follow-up once per-source timestamps are tracked centrally — the limits live
in ``source_registry.SOURCE_POLICIES`` and are attached here for context.
"""
from __future__ import annotations

from typing import Any

from .source_registry import SOURCE_POLICIES

# Live source modes that mean the live feed is not trustworthy.
_DEGRADED_MODES = {"estimated", "synthetic", "unavailable", "recording_pending"}


def _alert(source: str, status: str, severity: str, message: str) -> dict[str, Any]:
    policy = SOURCE_POLICIES.get(source)
    return {
        "source": source,
        "status": status,        # missing | fallback | degraded | unavailable | stale
        "severity": severity,    # info | warn | error
        "message": message,
        "freshness_seconds": policy.freshness_seconds if policy else None,
    }


def build_freshness_alerts(
    *,
    features: dict[str, Any] | None = None,
    last_run: dict[str, Any] | None = None,
    storage_health: dict[str, Any] | None = None,
    service: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return structured staleness/missing/fallback alerts from the current signals.

    Pure function — every input is a plain dict, so it is trivially testable and can
    be called on demand (no prediction run needed)."""
    features = features or {}
    last_run = last_run or {}
    storage_health = storage_health or {}
    service = service or {}
    alerts: list[dict[str, Any]] = []

    # Official results / Jolpica — the spine of every prediction.
    if not features.get("drivers"):
        alerts.append(_alert("official_results", "missing", "error",
                             "Prediction features unavailable — Jolpica/official results not loaded."))

    # Weather.
    weather_by_round = features.get("weather_by_round") or {}
    if not weather_by_round:
        alerts.append(_alert("open_meteo_session_weather", "missing", "warn",
                             "No session weather coverage loaded."))
    else:
        fallback_races = sum(1 for item in weather_by_round.values() if (item or {}).get("missing_data"))
        if fallback_races:
            alerts.append(_alert("weather_fallback", "fallback", "warn",
                                 f"Weather on neutral fallback for {fallback_races} race(s)."))

    # Sentiment (RSS/news).
    sentiment = features.get("sentiment") or {}
    if not ((sentiment.get("drivers") or {}) or (sentiment.get("teams") or {})):
        alerts.append(_alert("f1_sentiment", "stale", "info",
                             "No race-aware sentiment available (RSS/news)."))

    # Live source mode (from the most recent live run).
    mode = str(last_run.get("source_mode") or "").lower()
    if mode in _DEGRADED_MODES:
        ceiling = last_run.get("confidence_ceiling")
        alerts.append(_alert("live_session_engine", "degraded",
                             "error" if mode == "unavailable" else "warn",
                             f"Live source '{mode}'"
                             + (f" — confidence capped at {ceiling}." if ceiling is not None else ".")))
    if last_run.get("fallback_reason"):
        alerts.append(_alert("live_session_engine", "fallback", "warn",
                             f"Live fallback active: {last_run.get('fallback_reason')}."))

    # Bridges / storage.
    if not (storage_health.get("redis") or {}).get("available", False):
        alerts.append(_alert("redis", "unavailable", "error",
                             "Redis bridge unavailable — live probability/ops fan-out is dormant."))
    if not (storage_health.get("clickhouse") or {}).get("available", False):
        alerts.append(_alert("clickhouse", "unavailable", "warn",
                             "ClickHouse unavailable — metrics/odds history not persisting."))

    # Service liveness.
    if service.get("predictor_loaded") is False:
        alerts.append(_alert("model", "unavailable", "error", "Predictor not loaded."))
    if service.get("openf1") is False:
        alerts.append(_alert("openf1_session_facts", "unavailable", "warn",
                             "OpenF1 client unavailable — live timing/telemetry degraded."))

    return alerts


def alert_key(alert: dict[str, Any]) -> str:
    """Stable identity for an alert, for transition (new/resolved) detection."""
    return f"{alert.get('source')}:{alert.get('status')}"


def summarize_alerts(alerts: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"total": len(alerts), "error": 0, "warn": 0, "info": 0}
    for a in alerts:
        sev = a.get("severity")
        if sev in summary:
            summary[sev] += 1
    return summary
