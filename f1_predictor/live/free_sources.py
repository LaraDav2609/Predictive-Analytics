"""Free-first live F1 source normalization.

This module deliberately avoids paid/login-protected feeds. It combines the
free OpenF1 REST shape already used by the project, optional locally recorded
FastF1 timing files, and deterministic estimates so the dashboard always has a
truth-labeled live state to render.
"""

from __future__ import annotations

import json
import math
import os
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.openf1_client import offset_for_position, point_at_progress, progress_from_xy
from models.f1 import Driver, Race


SOURCE_CHAIN = [
    "openf1_rest_live_or_recent",
    "openf1_recent_historical",
    "fastf1_recorded_file",
    "estimated_minisector",
]


def load_fastf1_recorded_state(race: Race, drivers: list[Driver], session: str) -> dict[str, Any]:
    """Read an optional local recording and normalize the newest driver rows.

    The parser is intentionally permissive. It accepts JSON arrays/objects and
    JSONL-style files that contain rows with fields such as driver_number,
    racing_number, position, lap_number, gap_to_leader, interval, compound.
    """

    base = os.getenv("F1_FASTF1_LIVE_DIR")
    if not base:
        base = str(Path(__file__).resolve().parents[2] / "data" / "live_timing")
    directory = Path(base)
    if not directory.exists():
        return {"ok": False, "source": "fastf1_recorded_file", "reason": "recording_directory_missing", "drivers": {}}

    round_tokens = {f"round_{race.round}", f"round_{int(race.round):02d}", f"r{race.round}"}
    race_token = str(race.name or "").lower().replace(" ", "_")
    session_token = str(session or "race").lower()
    candidates = [
        item for item in directory.glob("*")
        if item.is_file()
        and item.suffix.lower() in {".json", ".jsonl", ".txt"}
        and session_token in item.name.lower()
        and (
            any(token and token in item.name.lower() for token in round_tokens)
            or bool(race_token and race_token in item.name.lower())
        )
    ]
    if not candidates:
        return {"ok": False, "source": "fastf1_recorded_file", "reason": "recording_file_missing", "drivers": {}}

    newest = max(candidates, key=lambda item: item.stat().st_mtime)
    rows = _read_recording_rows(newest)
    if not rows:
        return {
            "ok": False,
            "source": "fastf1_recorded_file",
            "reason": "recording_empty_or_unparsed",
            "path": str(newest),
            "file_size": newest.stat().st_size,
            "drivers": {},
        }

    driver_by_number = {int(driver.number): driver for driver in drivers if driver.number is not None}
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        number = _driver_number(row)
        if number is None:
            continue
        current = latest.setdefault(number, {})
        _deep_merge(current, row)

    normalized = {}
    for number, row in latest.items():
        driver = driver_by_number.get(number)
        if not driver:
            continue
        stints = _normalise_stints(_first(row, "stints", "Stints"))
        current_stint = _current_stint(stints)
        normalized[str(number)] = {
            "driver_number": number,
            "driver_id": driver.id,
            "driver_code": driver.code,
            "position": _safe_int(_first(row, "position", "Position", "position_current")),
            "gap_to_leader": _first(row, "gap_to_leader", "GapToLeader", "gap"),
            "interval": _interval_value(_first(row, "interval", "IntervalToPositionAhead", "interval_to_position_ahead")),
            "laps": _safe_int(_first(row, "lap_number", "LapNumber", "NumberOfLaps", "lap", "laps")),
            "best_lap": _lap_value(_first(row, "best_lap", "BestLapTime", "PersonalBestLapTime")),
            "representative_lap": _lap_value(_first(row, "representative_lap", "LastLapTime", "LapTime")),
            "pit_stops": _safe_int(_first(row, "pit_stops", "NumberOfPitStops")),
            "stints": stints,
            "compound": _first(row, "compound", "Compound", "tyre_compound") or current_stint.get("compound"),
            "tyre_age": current_stint.get("total_laps"),
            "tyre_new": current_stint.get("new"),
            "stint_number": current_stint.get("stint_number"),
            "date": _first(row, "date", "timestamp", "Utc", "utc"),
        }

    return {
        "ok": bool(normalized),
        "source": "fastf1_recorded_file",
        "path": str(newest),
        "file_size": newest.stat().st_size,
        "drivers": normalized,
        "reason": None if normalized else "recording_has_no_matching_drivers",
    }


def enrich_free_live_state(
    state: dict[str, Any],
    race: Race,
    drivers: list[Driver],
    openf1_session: dict[str, Any],
    track: dict[str, Any],
    recorded: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recorded = recorded or {}
    raw_counts = openf1_session.get("raw_counts") or {}
    track_points = track.get("racing_line") or track.get("display_points") or track.get("points") or []
    track_path = track.get("path") or ""
    recorded_rows = recorded.get("drivers") or {}
    session_schedule = _session_schedule(race, str(state.get("session") or "race"))
    if recorded.get("ok"):
        session_schedule = {**session_schedule, "status": "live_recorded", "is_future": False, "seconds_until_start": 0.0}

    source_mode = _source_mode(state, openf1_session, track, recorded)
    confidence = _source_confidence(source_mode, raw_counts, state, recorded_rows)
    fallback_reason = _fallback_reason(source_mode, openf1_session, track, recorded)
    last_success = _last_success(source_mode)
    collector_hint = _collector_hint(source_mode, recorded, session_schedule)

    source_rows = state.get("drivers") or _base_driver_rows(drivers)
    enriched_drivers = []
    for index, item in enumerate(source_rows):
        number_key = str(item.get("driver_number") or "")
        recorded_item = recorded_rows.get(number_key) or {}
        merged = {**item}
        if source_mode == "recorded":
            for key in ["position", "gap_to_leader", "interval", "laps"]:
                if recorded_item.get(key) is not None:
                    merged[key] = recorded_item.get(key)
            for key in ["best_lap", "representative_lap", "pit_stops", "stints", "tyre_age", "tyre_new", "stint_number"]:
                if recorded_item.get(key) is not None:
                    merged[key] = recorded_item.get(key)
            if recorded_item.get("compound"):
                merged["compounds"] = [recorded_item.get("compound")]
                merged["compound"] = recorded_item.get("compound")
        position = _safe_int(merged.get("position")) or index + 1
        laps = _safe_int(merged.get("laps")) or 0
        progress = _progress_for_driver(position, index, laps, len(drivers), source_mode)
        if merged.get("x") is not None and merged.get("y") is not None:
            progress = progress_from_xy(track, float(merged.get("x")), float(merged.get("y")))
        point = _point_from_progress(progress, track, position, len(drivers))
        merged["x"] = point["x"]
        merged["y"] = point["y"]
        merged["estimated_progress"] = round(progress, 4)
        merged["source_mode"] = _driver_source_mode(merged, source_mode)
        merged["confidence"] = _driver_confidence(merged["source_mode"], confidence)
        merged["is_estimated"] = merged["source_mode"] == "estimated"
        merged["data_status"] = merged["source_mode"]
        enriched_drivers.append(merged)

    enriched_drivers.sort(key=lambda row: row.get("position") or 99)
    by_driver_id = {item["driver_id"]: item for item in enriched_drivers if item.get("driver_id")}
    live_positions = {
        item["driver_id"]: {
            "position": item.get("position"),
            "x": item.get("x"),
            "y": item.get("y"),
            "gap_to_leader": item.get("gap_to_leader"),
            "interval": item.get("interval"),
            "representative_lap": item.get("representative_lap"),
            "compound": item.get("compound"),
            "tyre_age": item.get("tyre_age"),
            "tyre_new": item.get("tyre_new"),
            "pit_stops": item.get("pit_stops"),
            "estimated_progress": item.get("estimated_progress"),
            "source_mode": item.get("source_mode"),
            "confidence": item.get("confidence"),
            "is_estimated": item.get("is_estimated"),
        }
        for item in enriched_drivers
        if item.get("driver_id")
    }

    return {
        **state,
        "ok": source_mode != "unavailable",
        "mode": source_mode,
        "status": source_mode,
        "source_mode": source_mode,
        "source_chain": _source_chain(source_mode, recorded),
        "confidence": confidence,
        "last_successful_source": last_success,
        "fallback_reason": fallback_reason,
        "collector_hint": collector_hint,
        "session_schedule": session_schedule,
        "is_estimated": source_mode == "estimated",
        "drivers": enriched_drivers,
        "by_driver_id": by_driver_id,
        "live_positions": live_positions,
        "leader": enriched_drivers[0]["driver_id"] if enriched_drivers else None,
        "track": {
            **(state.get("track") or {}),
            "path": track_path or (state.get("track") or {}).get("path"),
            "track_key": track.get("track_key") or (state.get("track") or {}).get("track_key"),
            "points": track.get("points") or track_points,
            "display_points": track.get("display_points") or track_points,
            "racing_line": track_points,
            "geometry_version": track.get("geometry_version"),
            "mapping_source": track.get("mapping_source"),
            "geometry_confidence": track.get("geometry_confidence"),
            "markers": track.get("markers"),
            "sectors": track.get("sectors"),
            "drs_zones": track.get("drs_zones"),
            "pit_entry": track.get("pit_entry"),
            "pit_exit": track.get("pit_exit"),
            "source_mode": source_mode,
        },
        "sources": {
            **(state.get("sources") or {}),
            "openf1_session": openf1_session.get("source"),
            "openf1_mode": "live_or_recent" if openf1_session.get("ok") else "unavailable",
            "fastf1_recorded_file": "available" if recorded.get("ok") else recorded.get("reason") or "unavailable",
            "fastf1_recorded_path": recorded.get("path"),
            "fastf1_recorded_file_size": recorded.get("file_size"),
            "estimated_minisector": "available",
        },
    }


def source_diagnostics(state: dict[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    return {
        "ok": bool(state),
        "round": state.get("round"),
        "session": state.get("session"),
        "source_mode": state.get("source_mode") or state.get("mode") or "unavailable",
        "source_chain": state.get("source_chain") or SOURCE_CHAIN,
        "confidence": state.get("confidence") or 0.0,
        "last_successful_source": state.get("last_successful_source"),
        "fallback_reason": state.get("fallback_reason") or state.get("reason"),
        "collector_hint": state.get("collector_hint"),
        "session_schedule": state.get("session_schedule") or {},
        "is_estimated": bool(state.get("is_estimated")),
        "data_age_seconds": state.get("data_age_seconds") or state.get("age_seconds"),
        "sources": state.get("sources") or {},
        "driver_count": len(state.get("drivers") or []),
    }


def _read_recording_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return []
    if not text:
        return []
    try:
        data = json.loads(text)
        return _flatten_records(data)
    except ValueError:
        try:
            return _flatten_records(ast.literal_eval(text))
        except (ValueError, SyntaxError):
            pass
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.extend(_flatten_records(json.loads(line)))
            except ValueError:
                try:
                    rows.extend(_flatten_records(ast.literal_eval(line)))
                except (ValueError, SyntaxError):
                    continue
        return rows


def _flatten_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], str):
            payload = value[1]
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    try:
                        payload = ast.literal_eval(payload)
                    except (ValueError, SyntaxError):
                        payload = None
            if isinstance(payload, (dict, list)):
                rows = _flatten_records(payload)
                timestamp = value[2] if len(value) > 2 else None
                for row in rows:
                    row.setdefault("category", value[0])
                    if timestamp:
                        row.setdefault("timestamp", timestamp)
                return rows
        rows = []
        for item in value:
            rows.extend(_flatten_records(item))
        return rows
    if isinstance(value, dict):
        if value and all(str(key).isdigit() for key in value.keys()):
            rows = []
            for key, item in value.items():
                if isinstance(item, dict):
                    rows.append({"driver_number": int(key), **item})
            if rows:
                return rows
        rows = []
        for key in ["data", "Data", "cars", "Cars", "timing", "TimingData", "lines", "Lines"]:
            child = value.get(key)
            if isinstance(child, (list, dict)):
                rows.extend(_flatten_records(child))
        return rows or [value]
    return []


def _deep_merge(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if value in (None, ""):
            continue
        if key == "Stints" and isinstance(target.get(key), list) and isinstance(value, dict):
            target[key] = _merge_stint_updates(target[key], value)
            continue
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _merge_stint_updates(existing: list[Any], updates: dict[str, Any]) -> list[Any]:
    merged = [dict(item) if isinstance(item, dict) else item for item in existing]
    for key, value in updates.items():
        index = _safe_int(key)
        if index is None:
            continue
        while len(merged) <= index:
            merged.append({})
        if isinstance(merged[index], dict) and isinstance(value, dict):
            _deep_merge(merged[index], value)
        else:
            merged[index] = value
    return merged


def _normalise_stints(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    if isinstance(value, dict):
        iterable = sorted(value.items(), key=lambda item: _safe_int(item[0]) or 0)
        items = [(idx, item) for idx, item in iterable if isinstance(item, dict)]
    elif isinstance(value, list):
        items = [(idx, item) for idx, item in enumerate(value) if isinstance(item, dict)]
    else:
        return []

    stints = []
    for idx, item in items:
        stint_number = _safe_int(idx)
        stints.append({
            "stint_number": (stint_number + 1) if stint_number is not None else len(stints) + 1,
            "compound": item.get("Compound") or item.get("compound"),
            "new": _boolish(item.get("New") if "New" in item else item.get("new")),
            "total_laps": _safe_int(item.get("TotalLaps") if "TotalLaps" in item else item.get("total_laps")),
            "start_laps": _safe_int(item.get("StartLaps") if "StartLaps" in item else item.get("start_laps")),
            "lap_time": _lap_value(item.get("LapTime") if "LapTime" in item else item.get("lap_time")),
            "lap_number": _safe_int(item.get("LapNumber") if "LapNumber" in item else item.get("lap_number")),
        })
    return stints


def _current_stint(stints: list[dict[str, Any]]) -> dict[str, Any]:
    if not stints:
        return {}
    with_compound = [item for item in stints if item.get("compound")]
    return (with_compound or stints)[-1]


def _base_driver_rows(drivers: list[Driver]) -> list[dict[str, Any]]:
    return [
        {
            "driver_id": driver.id,
            "driver_number": int(driver.number) if driver.number is not None else None,
            "driver_code": driver.code,
            "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
            "team": driver.team,
            "position": index + 1,
            "gap_to_leader": None,
            "interval": None,
            "best_lap": None,
            "representative_lap": None,
            "median_lap": None,
            "laps": None,
            "stints": None,
            "compounds": [],
            "pit_stops": None,
            "x": None,
            "y": None,
            "data_status": "estimated",
        }
        for index, driver in enumerate(sorted(drivers, key=lambda item: item.points, reverse=True))
    ]


def _source_mode(state: dict[str, Any], openf1_session: dict[str, Any], track: dict[str, Any], recorded: dict[str, Any]) -> str:
    if state.get("mode") == "live":
        return "live"
    raw_counts = openf1_session.get("raw_counts") or {}
    if openf1_session.get("ok") and any(int(raw_counts.get(key) or 0) > 0 for key in ["positions", "laps", "intervals"]):
        return "recent"
    if recorded.get("ok"):
        return "recorded"
    if state.get("drivers") or track.get("path"):
        return "estimated"
    return "unavailable"


def _source_confidence(source_mode: str, raw_counts: dict[str, Any], state: dict[str, Any], recorded_rows: dict[str, Any]) -> float:
    if source_mode == "live":
        return 0.92
    if source_mode == "recent":
        count = sum(int(raw_counts.get(key) or 0) for key in ["positions", "laps", "intervals", "stints", "pits"])
        return round(min(0.78, 0.45 + count / 900.0), 3)
    if source_mode == "recorded":
        return round(min(0.70, 0.38 + len(recorded_rows) / 60.0), 3)
    if source_mode == "estimated":
        with_positions = len(state.get("live_positions") or {})
        return round(0.18 + min(0.12, with_positions / 200.0), 3)
    return 0.0


def _driver_source_mode(item: dict[str, Any], source_mode: str) -> str:
    if source_mode in {"live", "recorded"}:
        return source_mode
    if source_mode == "recent" and (item.get("position") is not None or item.get("representative_lap") is not None):
        return "recent"
    if item.get("x") is not None and item.get("y") is not None:
        return "estimated"
    return "unavailable"


def _driver_confidence(source_mode: str, base: float) -> float:
    if source_mode == "unavailable":
        return 0.0
    if source_mode == "estimated":
        return min(base, 0.30)
    return base


def _fallback_reason(source_mode: str, openf1_session: dict[str, Any], track: dict[str, Any], recorded: dict[str, Any]) -> str | None:
    if source_mode == "live":
        return None
    if source_mode == "recent":
        return "true_live_location_unavailable_using_recent_openf1_rest"
    if source_mode == "recorded":
        return "openf1_live_unavailable_using_local_fastf1_recording"
    if source_mode == "estimated":
        return openf1_session.get("reason") or track.get("reason") or recorded.get("reason") or "free_live_sources_unavailable_using_estimate"
    return openf1_session.get("reason") or track.get("reason") or recorded.get("reason") or "free_live_sources_unavailable"


def _collector_hint(source_mode: str, recorded: dict[str, Any], schedule: dict[str, Any]) -> str | None:
    if source_mode in {"live", "recent", "recorded"}:
        return None
    reason = recorded.get("reason")
    seconds_until = schedule.get("seconds_until_start")
    if reason == "recording_empty_or_unparsed" and isinstance(seconds_until, (int, float)) and seconds_until > 0:
        return "collector_ready_session_not_live_yet"
    if reason == "recording_empty_or_unparsed":
        return "collector_file_empty_waiting_for_timing_messages"
    if reason == "recording_file_missing":
        return "start_collector_to_record_free_fastf1_timing"
    return None


def _last_success(source_mode: str) -> str | None:
    return {
        "live": "openf1_rest_live_or_recent",
        "recent": "openf1_recent_historical",
        "recorded": "fastf1_recorded_file",
        "estimated": "estimated_minisector",
    }.get(source_mode)


def _session_schedule(race: Race, session: str) -> dict[str, Any]:
    session_key = (session or "race").lower()
    selected = None
    for item in race.sessions or []:
        code = str(item.get("code") or "").lower()
        if code == session_key or (session_key == "qualifying" and code.startswith("qual")):
            selected = item
            break
    if selected is None and session_key == "race":
        selected = {"code": "race", "name": "Race", "date": race.date.isoformat(), "status": race.status.lower()}
    elif selected is None:
        selected = {"code": session_key, "name": session_key.title(), "date": None, "status": "unknown"}

    start = _parse_datetime(selected.get("date"))
    now = datetime.now(timezone.utc)
    seconds_until = None
    if start is not None:
        seconds_until = round((start - now).total_seconds(), 3)
    return {
        "code": selected.get("code") or session_key,
        "name": selected.get("name") or session_key.title(),
        "scheduled_start": start.isoformat() if start else None,
        "status": selected.get("status") or "unknown",
        "seconds_until_start": seconds_until,
        "is_future": bool(seconds_until is not None and seconds_until > 0),
    }


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _source_chain(source_mode: str, recorded: dict[str, Any]) -> list[dict[str, Any]]:
    reached = _last_success(source_mode)
    rows = []
    for name in SOURCE_CHAIN:
        rows.append({
            "name": name,
            "status": "used" if name == reached else ("checked" if reached else "unavailable"),
            "reason": recorded.get("reason") if name == "fastf1_recorded_file" and not recorded.get("ok") else None,
        })
        if name == reached:
            break
    return rows


def _progress_for_driver(position: int, index: int, laps: int, field_size: int, source_mode: str) -> float:
    field = max(1, field_size)
    base = ((laps % 70) / 70.0) if laps else 0.18
    offset = (max(0, position - 1) / field) * 0.055
    if source_mode == "estimated":
        offset += index * 0.012
    return (base - offset) % 1.0


def _point_from_progress(progress: float, track: dict[str, Any], position: int, field_size: int) -> dict[str, float]:
    if track:
        return point_at_progress(track, progress, offset_for_position(position, field_size))
    angle = progress * math.tau
    lane_offset = (position - 1) / max(1, field_size) * 16.0
    return {
        "x": round(500 + math.cos(angle) * (360 - lane_offset), 2),
        "y": round(215 + math.sin(angle) * (145 - lane_offset * 0.3), 2),
    }


def _row_time(row: dict[str, Any]) -> str:
    return str(_first(row, "date", "timestamp", "Utc", "utc", "Time") or "")


def _driver_number(row: dict[str, Any]) -> int | None:
    return _safe_int(_first(row, "driver_number", "RacingNumber", "racing_number", "number", "Number"))


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _interval_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("Value") or value.get("value")
    return value


def _lap_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("Value") or value.get("value")
    return value


def _boolish(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "new"}


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
