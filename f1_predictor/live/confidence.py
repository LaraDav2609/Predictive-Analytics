"""Evidence-based confidence policy for F1 live prediction."""

from __future__ import annotations

from typing import Any


SOURCE_CEILINGS = {
    "unavailable": 0.05,
    "estimated": 0.25,
    "recent": 0.55,
    "historical": 0.55,
    "recorded": 0.75,
    "live": 0.90,
}
FULL_LIVE_CEILING = 0.95
SOURCE_FLOORS = {
    "unavailable": 0.0,
    "estimated": 0.08,
    "recent": 0.22,
    "historical": 0.22,
    "recorded": 0.38,
    "live": 0.50,
}


def build_live_confidence_report(
    truth: dict[str, Any] | None,
    *,
    live_state: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    recorder_status: dict[str, Any] | None = None,
    probabilities: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    truth = truth or {}
    live_state = live_state or {}
    diagnostics = diagnostics or {}
    recorder_status = recorder_status or {}
    probabilities = probabilities or []

    source_mode = str(truth.get("source_mode") or live_state.get("source_mode") or live_state.get("mode") or "unavailable").lower()
    rows = _drivers(truth, live_state)
    coverage = _coverage_counts(rows, truth, live_state, diagnostics)
    openf1_counts = _openf1_counts(truth, live_state, diagnostics)
    missing = _missing_groups(source_mode, coverage, truth, recorder_status)
    blockers = _blockers(source_mode, coverage, truth, recorder_status, probabilities)
    ceiling = _confidence_ceiling(source_mode, coverage)
    raw_confidence = _raw_confidence(source_mode, truth, live_state, coverage, blockers)
    floor = SOURCE_FLOORS.get(source_mode, 0.0)
    confidence = round(max(floor, min(ceiling, raw_confidence)), 4)

    return {
        "ok": True,
        "source_mode": source_mode,
        "confidence": confidence,
        "confidence_ceiling": ceiling,
        "confidence_reason": _confidence_reason(source_mode, confidence, ceiling, blockers, coverage),
        "coverage_counts": coverage,
        "openf1_row_counts": openf1_counts,
        "missing_confidence_groups": missing,
        "confidence_blockers": blockers,
        "next_best_action": _next_best_action(source_mode, missing, recorder_status, live_state),
        "recorder_status": recorder_status,
    }


def attach_confidence_report(payload: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return payload
    payload["confidence"] = report.get("confidence", payload.get("confidence"))
    payload["confidence_ceiling"] = report.get("confidence_ceiling")
    payload["confidence_reason"] = report.get("confidence_reason")
    payload["coverage_counts"] = report.get("coverage_counts") or {}
    payload["openf1_row_counts"] = report.get("openf1_row_counts") or {}
    payload["missing_confidence_groups"] = report.get("missing_confidence_groups") or []
    payload["confidence_blockers"] = report.get("confidence_blockers") or []
    payload["next_best_action"] = report.get("next_best_action")
    payload["confidence_report"] = report
    return payload


def _drivers(truth: dict[str, Any], live_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = truth.get("drivers")
    if isinstance(rows, list) and rows:
        return rows
    rows = live_state.get("drivers")
    if isinstance(rows, list) and rows:
        return rows
    positions = live_state.get("live_positions") or truth.get("live_positions") or {}
    if isinstance(positions, dict):
        return [{"driver_id": driver_id, **row} for driver_id, row in positions.items()]
    return []


def _coverage_counts(
    rows: list[dict[str, Any]],
    truth: dict[str, Any],
    live_state: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    total = len(rows)
    real_rows = [
        row for row in rows
        if str(row.get("source_mode") or truth.get("source_mode") or "").lower() not in {"estimated", "unavailable"}
    ]
    weather = (truth.get("signals") or {}).get("weather") or live_state.get("weather") or {}
    race_control = (truth.get("signals") or {}).get("race_control") or live_state.get("race_control") or {}
    raw_counts = _openf1_counts(truth, live_state, diagnostics)
    return {
        "driver_count": total,
        "usable_driver_count": len(real_rows),
        "real_positions": _count(rows, "position", real_only=True),
        "real_gaps": _count(rows, "gap_to_leader", real_only=True),
        "real_intervals": _count(rows, "interval", real_only=True),
        "lap_count": _count(rows, "lap", real_only=True),
        "pace_count": _count_any(rows, ["representative_lap", "best_lap", "median_lap"], real_only=True),
        "tyre_count": _count_any(rows, ["compound", "tyre_age"], real_only=True),
        "pit_count": _count(rows, "pit_stops", real_only=True),
        "weather_available": bool(weather) and not weather.get("missing_data"),
        "race_control_available": bool(race_control) or int(raw_counts.get("race_control") or 0) > 0,
        "location_count": _count_any(rows, ["x", "y", "progress", "estimated_progress"], real_only=True),
    }


def _openf1_counts(truth: dict[str, Any], live_state: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, int]:
    raw = (
        ((live_state.get("signals") or {}).get("raw_counts") or {})
        or ((diagnostics.get("openf1") or {}).get("raw_counts") or {})
        or ((truth.get("signals") or {}).get("raw_counts") or {})
    )
    aliases = {
        "positions": ["positions", "position"],
        "intervals": ["intervals", "interval"],
        "laps": ["laps", "lap"],
        "stints": ["stints", "stint"],
        "pits": ["pits", "pit"],
        "weather": ["weather"],
        "race_control": ["race_control", "raceControl"],
        "locations": ["locations", "location", "car_data"],
    }
    result: dict[str, int] = {}
    for name, keys in aliases.items():
        result[name] = max(_as_int(raw.get(key)) for key in keys)
    return result


def _missing_groups(
    source_mode: str,
    coverage: dict[str, Any],
    truth: dict[str, Any],
    recorder_status: dict[str, Any],
) -> list[str]:
    missing = set(str(item) for item in (truth.get("missing_groups") or []))
    if coverage["usable_driver_count"] < 18:
        missing.add("usable_live_timing_rows")
    if coverage["real_positions"] < 18:
        missing.add("real_running_order")
    if coverage["real_gaps"] < 12 and coverage["real_intervals"] < 12:
        missing.add("real_gaps_intervals")
    if coverage["tyre_count"] < 12:
        missing.add("tyres_stints")
    if coverage["pit_count"] < 12:
        missing.add("pit_stops")
    if not coverage["weather_available"]:
        missing.add("session_weather")
    if not coverage["race_control_available"]:
        missing.add("race_control")
    if source_mode == "estimated" and not recorder_status.get("running") and not recorder_status.get("file_size"):
        missing.add("fastf1_recording")
    return sorted(missing)


def _blockers(
    source_mode: str,
    coverage: dict[str, Any],
    truth: dict[str, Any],
    recorder_status: dict[str, Any],
    probabilities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if source_mode in {"estimated", "unavailable"}:
        blockers.append({"code": "estimated_source", "message": "No real live timing source is active.", "severity": "high"})
    if coverage["usable_driver_count"] < 18:
        blockers.append({"code": "thin_driver_rows", "message": "Fewer than 18 drivers have usable timing rows.", "severity": "high"})
    if coverage["real_gaps"] < 12 and coverage["real_intervals"] < 12:
        blockers.append({"code": "missing_gaps", "message": "Gaps and intervals are missing or estimated.", "severity": "medium"})
    if coverage["tyre_count"] < 12:
        blockers.append({"code": "missing_tyres", "message": "Tyre compound or stint age is not available for most cars.", "severity": "medium"})
    if not coverage["race_control_available"]:
        blockers.append({"code": "missing_race_control", "message": "Race-control status is unavailable.", "severity": "low"})
    if truth.get("source_disagreement"):
        blockers.append({"code": "source_disagreement", "message": "Source disagreement detected; confidence is flattened.", "severity": "medium"})
    top = max([float(row.get("win_probability") or row.get("calibrated_probability") or 0.0) for row in probabilities] or [0.0])
    if top >= 0.45 and source_mode in {"estimated", "unavailable", "recent"}:
        blockers.append({"code": "overconfidence_guard", "message": "Top probability is high for the current source quality.", "severity": "medium"})
    if source_mode == "estimated" and not recorder_status.get("fastf1_available"):
        blockers.append({"code": "fastf1_unavailable", "message": "FastF1 recorder is not available in the dashboard runtime.", "severity": "medium"})
    return blockers[:8]


def _confidence_ceiling(source_mode: str, coverage: dict[str, Any]) -> float:
    ceiling = SOURCE_CEILINGS.get(source_mode, SOURCE_CEILINGS["estimated"])
    if source_mode == "live":
        full_live = (
            coverage["real_positions"] >= 18
            and (coverage["real_gaps"] >= 12 or coverage["real_intervals"] >= 12)
            and coverage["tyre_count"] >= 12
            and coverage["weather_available"]
            and coverage["race_control_available"]
        )
        if full_live:
            ceiling = FULL_LIVE_CEILING
    return round(ceiling, 4)


def _raw_confidence(
    source_mode: str,
    truth: dict[str, Any],
    live_state: dict[str, Any],
    coverage: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> float:
    base = _as_float(truth.get("confidence"), _as_float(live_state.get("confidence"), 0.0))
    if base <= 0:
        base = {
            "live": 0.78,
            "recorded": 0.62,
            "recent": 0.46,
            "historical": 0.42,
            "estimated": 0.18,
        }.get(source_mode, 0.02)
    evidence = (
        min(1.0, coverage["usable_driver_count"] / 20.0) * 0.24
        + min(1.0, coverage["real_positions"] / 20.0) * 0.20
        + min(1.0, max(coverage["real_gaps"], coverage["real_intervals"]) / 20.0) * 0.18
        + min(1.0, coverage["tyre_count"] / 20.0) * 0.14
        + min(1.0, coverage["pit_count"] / 20.0) * 0.08
        + (0.08 if coverage["weather_available"] else 0.0)
        + (0.08 if coverage["race_control_available"] else 0.0)
    )
    if source_mode in {"estimated", "unavailable"}:
        evidence *= 0.35
    severity_penalty = sum({"high": 0.08, "medium": 0.045, "low": 0.02}.get(item.get("severity"), 0.025) for item in blockers)
    return max(0.0, base * 0.55 + evidence * 0.45 - severity_penalty)


def _confidence_reason(
    source_mode: str,
    confidence: float,
    ceiling: float,
    blockers: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> str:
    if blockers:
        return blockers[0]["message"]
    if confidence >= ceiling - 0.02:
        return f"{source_mode} source is near its confidence ceiling."
    if coverage["usable_driver_count"] >= 18:
        return "Most timing groups are present; confidence is evidence-backed."
    return "Confidence is limited by incomplete live timing coverage."


def _next_best_action(
    source_mode: str,
    missing: list[str],
    recorder_status: dict[str, Any],
    live_state: dict[str, Any],
) -> str:
    if source_mode == "live":
        return "Monitor refresh stability and race-control/weather events."
    if source_mode == "recorded":
        return "Keep recorder running and verify file growth."
    if "fastf1_recording" in missing:
        if recorder_status.get("fastf1_available"):
            return "Start FastF1 recorder before or during the live session."
        return "Install or configure FastF1 Python runtime for local recording."
    if source_mode == "recent":
        return "Wait for fresher OpenF1 timing rows or start local recorder."
    if live_state.get("fallback_reason"):
        return f"Resolve fallback: {live_state.get('fallback_reason')}"
    return "Waiting for OpenF1 timing rows."


def _count(rows: list[dict[str, Any]], key: str, *, real_only: bool = False) -> int:
    return sum(1 for row in rows if row.get(key) is not None and (not real_only or _real(row)))


def _count_any(rows: list[dict[str, Any]], keys: list[str], *, real_only: bool = False) -> int:
    return sum(1 for row in rows if any(row.get(key) is not None for key in keys) and (not real_only or _real(row)))


def _real(row: dict[str, Any]) -> bool:
    return str(row.get("source_mode") or "").lower() not in {"estimated", "unavailable"}


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
