"""Source-aware race/session truth snapshots.

This module merges the best available race facts into a single shape that can
be used by API routes, simulation, storage, and dashboard rendering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from models.f1 import Driver, Race


def build_race_truth_snapshot(
    race: Race,
    drivers: list[Driver],
    session: str = "race",
    profile: dict[str, Any] | None = None,
    openf1_session: dict[str, Any] | None = None,
    live_state: dict[str, Any] | None = None,
    weather: dict[str, Any] | None = None,
    live: bool = False,
) -> dict[str, Any]:
    """Build a normalized, source-labeled race truth snapshot."""

    session_key = _session_kind(session)
    profile = profile or {}
    openf1_session = openf1_session or {}
    live_state = live_state or {}
    now = datetime.now(timezone.utc).isoformat()

    source_chain = [
        "f1_client_official_results",
        "openf1_session_facts",
        "live_session_engine",
        "fastf1_recorded_file",
        "estimated_standings_order",
    ]
    official_rows = _official_rows(profile, session_key)
    openf1_rows = _openf1_rows(openf1_session, drivers)
    live_rows = _live_rows(live_state)

    mode, confidence, fallback_reason = _snapshot_quality(
        race=race,
        live=live,
        live_state=live_state,
        openf1_rows=openf1_rows,
        official_rows=official_rows,
        session=session_key,
    )

    by_driver: dict[str, dict[str, Any]] = {}
    for index, driver in enumerate(drivers, start=1):
        by_driver[driver.id] = {
            "driver_id": driver.id,
            "driver_number": driver.number,
            "driver_code": driver.code,
            "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
            "team": driver.team,
            "position": driver.position or index,
            "championship_position": driver.position,
            "gap_to_leader": None,
            "interval": None,
            "lap": None,
            "laps": None,
            "best_lap": None,
            "representative_lap": None,
            "compound": None,
            "tyre_age": None,
            "pit_stops": None,
            "estimated_progress": _estimated_progress(index, len(drivers)),
            "x": None,
            "y": None,
            "points": driver.points,
            "result_status": None,
            "source_mode": "estimated",
            "confidence": 0.25,
            "is_estimated": True,
        }

    _merge_rows(by_driver, official_rows, "historical", 0.82)
    _merge_rows(by_driver, openf1_rows, "recent", 0.68)
    if live and live_rows:
        _merge_rows(by_driver, live_rows, mode, confidence)

    rows = sorted(by_driver.values(), key=lambda item: item.get("position") or 99)
    missing_groups = _missing_groups(mode, official_rows, openf1_session, live_state, session_key, weather)
    signals = _signals(openf1_session, live_state, weather)
    return {
        "ok": True,
        "round": race.round,
        "race_name": race.name,
        "session": session_key,
        "status": _status(race, session_key, official_rows, live, mode),
        "source_mode": mode,
        "source_chain": source_chain,
        "confidence": round(confidence, 4),
        "data_age_seconds": _age_seconds(live_state),
        "last_successful_source": _last_successful_source(mode, live_state, bool(openf1_rows), bool(official_rows)),
        "fallback_reason": fallback_reason,
        "is_estimated": mode == "estimated",
        "missing_groups": missing_groups,
        "generated_at": now,
        "drivers": rows,
        "by_driver_id": {item["driver_id"]: item for item in rows},
        "live_positions": {
            item["driver_id"]: {
                "position": item.get("position"),
                "x": item.get("x"),
                "y": item.get("y"),
                "gap_to_leader": item.get("gap_to_leader"),
                "interval": item.get("interval"),
                "representative_lap": item.get("representative_lap"),
                "compound": item.get("compound"),
                "tyre_age": item.get("tyre_age"),
                "pit_stops": item.get("pit_stops"),
                "estimated_progress": item.get("estimated_progress"),
                "source_mode": item.get("source_mode"),
                "confidence": item.get("confidence"),
                "is_estimated": item.get("is_estimated"),
            }
            for item in rows
        },
        "signals": signals,
        "sources": {
            "official_results": "available" if official_rows else "unavailable",
            "openf1_session": openf1_session.get("source") if openf1_session.get("ok") else openf1_session.get("reason") or "unavailable",
            "live_state": live_state.get("source_mode") or live_state.get("mode") or "unavailable",
            "weather": "available" if signals.get("weather") else "unavailable",
            "race_control": "available" if signals.get("race_control") else "unavailable",
        },
    }


def _official_rows(profile: dict[str, Any], session: str) -> list[dict[str, Any]]:
    if session == "qualifying":
        rows = profile.get("qualifying") or []
    elif session == "sprint":
        rows = profile.get("sprint") or []
    else:
        rows = profile.get("results") or []
    return [row for row in rows if row.get("driver_id")]


def _openf1_rows(openf1_session: dict[str, Any], drivers: list[Driver]) -> list[dict[str, Any]]:
    if not openf1_session.get("ok"):
        return []
    by_number = {str(driver.number): driver.id for driver in drivers if driver.number is not None}
    positions = ((openf1_session.get("positions") or {}).get("drivers") or {})
    intervals = ((openf1_session.get("intervals") or {}).get("drivers") or {})
    laps = ((openf1_session.get("laps") or {}).get("drivers") or {})
    stints = ((openf1_session.get("stints") or {}).get("drivers") or {})
    pits = ((openf1_session.get("pits") or {}).get("drivers") or {})
    rows = []
    for number, driver_id in by_number.items():
        position = positions.get(number) or {}
        lap = laps.get(number) or {}
        stint = stints.get(number) or {}
        pit = pits.get(number) or {}
        interval = intervals.get(number) or {}
        if not any([position, lap, stint, pit, interval]):
            continue
        compounds = stint.get("compounds") or []
        rows.append({
            "driver_id": driver_id,
            "position": _safe_int(position.get("position")),
            "gap_to_leader": interval.get("gap_to_leader"),
            "interval": interval.get("interval"),
            "lap": _safe_int(lap.get("lap")),
            "laps": _safe_int(lap.get("laps")),
            "best_lap": lap.get("best_lap"),
            "representative_lap": lap.get("representative_lap"),
            "compound": stint.get("compound") or (compounds[-1] if compounds else None),
            "tyre_age": _safe_int(stint.get("tyre_age") or stint.get("avg_stint_laps")),
            "pit_stops": _safe_int(pit.get("pit_stops")),
        })
    return rows


def _live_rows(live_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = live_state.get("drivers")
    if isinstance(rows, list) and rows:
        return rows
    live_positions = live_state.get("live_positions") or {}
    if isinstance(live_positions, dict):
        return [{"driver_id": driver_id, **payload} for driver_id, payload in live_positions.items()]
    return []


def _merge_rows(target: dict[str, dict[str, Any]], rows: list[dict[str, Any]], source_mode: str, confidence: float) -> None:
    for index, row in enumerate(rows, start=1):
        driver_id = row.get("driver_id")
        if not driver_id or driver_id not in target:
            continue
        current = target[driver_id]
        for key in [
            "position",
            "gap_to_leader",
            "interval",
            "lap",
            "laps",
            "best_lap",
            "representative_lap",
            "compound",
            "tyre_age",
            "pit_stops",
            "estimated_progress",
            "x",
            "y",
            "points",
            "status",
            "time",
        ]:
            value = row.get(key)
            if value is not None:
                current[key] = value
        if current.get("position") is None:
            current["position"] = index
        current["result_status"] = row.get("status") or current.get("result_status")
        current["source_mode"] = row.get("source_mode") or source_mode
        current["confidence"] = round(float(row.get("confidence") if row.get("confidence") is not None else confidence), 4)
        current["is_estimated"] = current["source_mode"] == "estimated"


def _snapshot_quality(
    race: Race,
    live: bool,
    live_state: dict[str, Any],
    openf1_rows: list[dict[str, Any]],
    official_rows: list[dict[str, Any]],
    session: str,
) -> tuple[str, float, str | None]:
    if live:
        mode = live_state.get("source_mode") or live_state.get("mode")
        if mode in {"live", "recorded", "recent", "historical", "estimated"}:
            return mode, float(live_state.get("confidence") or _mode_confidence(mode)), live_state.get("fallback_reason")
    if official_rows:
        return "historical", 0.84, None
    if openf1_rows:
        return "recent", 0.68, None
    if race.status == "COMPLETED":
        return "unavailable", 0.0, f"No {session} classification could be resolved for completed race"
    return "estimated", 0.28, "No live/recent session facts available; using standings-based estimated order"


def _signals(openf1_session: dict[str, Any], live_state: dict[str, Any], weather: dict[str, Any] | None = None) -> dict[str, Any]:
    open_weather = openf1_session.get("weather") or {}
    open_control = openf1_session.get("race_control") or {}
    live_signals = live_state.get("signals") or {}
    weather = live_signals.get("weather") or (open_weather if not open_weather.get("missing_data") else None) or weather or open_weather
    race_control = live_signals.get("race_control") or open_control
    chaos = max(float((weather or {}).get("chaos_score") or 0.0), float((race_control or {}).get("chaos_score") or 0.0), float(live_signals.get("chaos_score") or 0.0))
    return {
        "chaos_score": round(chaos, 4),
        "weather": weather,
        "race_control": race_control,
        "raw_counts": openf1_session.get("raw_counts") or live_signals.get("raw_counts") or {},
    }


def _missing_groups(
    mode: str,
    official_rows: list[dict[str, Any]],
    openf1_session: dict[str, Any],
    live_state: dict[str, Any],
    session: str,
    weather: dict[str, Any] | None = None,
) -> list[str]:
    missing = []
    if mode in {"estimated", "unavailable"}:
        missing.append("live_timing")
    if not official_rows and session in {"race", "qualifying", "sprint"}:
        missing.append(f"official_{session}_classification")
    raw_counts = openf1_session.get("raw_counts") or {}
    for key in ["positions", "intervals", "laps", "stints", "pits", "weather", "race_control"]:
        if key == "weather" and weather and not weather.get("missing_data"):
            continue
        if not raw_counts.get(key):
            missing.append(f"openf1_{key}")
    if not live_state:
        missing.append("live_state")
    return sorted(set(missing))


def _status(race: Race, session: str, official_rows: list[dict[str, Any]], live: bool, mode: str) -> str:
    if live and mode == "live":
        return "live"
    if official_rows:
        return f"{session}_classification_available"
    if mode in {"recent", "recorded", "historical"}:
        return f"{mode}_session_facts"
    if race.status == "COMPLETED":
        return "completed_missing_classification"
    return "pre_session_estimate"


def _last_successful_source(mode: str, live_state: dict[str, Any], has_openf1: bool, has_official: bool) -> str | None:
    if live_state.get("last_successful_source"):
        return live_state.get("last_successful_source")
    if mode == "recorded":
        return "fastf1_recorded_file"
    if mode in {"live", "recent"} or has_openf1:
        return "openf1_session_facts"
    if has_official:
        return "f1_client_official_results"
    if mode == "estimated":
        return "estimated_standings_order"
    return None


def _mode_confidence(mode: str) -> float:
    return {
        "live": 0.9,
        "recorded": 0.78,
        "recent": 0.66,
        "historical": 0.82,
        "estimated": 0.28,
        "unavailable": 0.0,
    }.get(mode, 0.25)


def _age_seconds(live_state: dict[str, Any]) -> float | None:
    value = live_state.get("data_age_seconds")
    if value is not None:
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None
    return None


def _estimated_progress(index: int, count: int) -> float:
    if count <= 0:
        return 0.0
    return round(((index - 1) / count) % 1.0, 4)


def _session_kind(session: str) -> str:
    value = (session or "race").lower()
    normalized = value.replace("_", " ").replace("-", " ")
    if normalized.startswith("fp") or "practice" in normalized:
        return "practice"
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint"):
        return "sprint"
    return "race"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
