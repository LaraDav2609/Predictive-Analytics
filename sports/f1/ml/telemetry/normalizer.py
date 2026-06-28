"""Normalize provider-specific telemetry rows into trace points."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sports.f1.ml.telemetry.types import TelemetryTracePoint


def normalize_openf1_trace_points(
    *,
    car_rows: list[dict[str, Any]],
    location_rows: list[dict[str, Any]] | None = None,
    race_id: str,
    session: str,
    driver_lookup: dict[int, str] | None = None,
    source: str = "openf1",
    max_location_gap_seconds: float = 1.25,
) -> list[TelemetryTracePoint]:
    """Join OpenF1 car data and location rows by nearest timestamp.

    OpenF1 exposes car channels and location as separate endpoints. For model
    features we only need a best-effort trace, so a small nearest-time join is
    enough and keeps the function deterministic for fixture tests.
    """

    driver_lookup = driver_lookup or {}
    location_index = _index_locations(location_rows or [])
    points: list[TelemetryTracePoint] = []
    for row in car_rows or []:
        number = _safe_int(_first(row, "driver_number", "DriverNumber", "number"))
        ts = _parse_datetime(_first(row, "date", "Date", "timestamp", "Time"))
        if ts is None:
            continue
        code = _driver_code(row, number, driver_lookup)
        location = _nearest_location(location_index.get(number or -1, []), ts, max_location_gap_seconds)
        points.append(
            TelemetryTracePoint(
                race_id=race_id,
                session=session,
                driver_code=code,
                driver_number=number,
                timestamp=ts,
                lap=_safe_int(_first(row, "lap_number", "lap", "LapNumber")),
                distance_m=_safe_float(_first(row, "distance", "Distance", "s_coord")),
                x=_safe_float(_first(location, "x", "X")) if location else None,
                y=_safe_float(_first(location, "y", "Y")) if location else None,
                z=_safe_float(_first(location, "z", "Z")) if location else None,
                speed_kph=_safe_float(_first(row, "speed", "Speed", "speed_kph")),
                throttle_pct=_safe_float(_first(row, "throttle", "Throttle")),
                brake_pct=_brake_pct(_first(row, "brake", "Brake")),
                rpm=_safe_int(_first(row, "rpm", "RPM")),
                gear=_safe_int(_first(row, "n_gear", "nGear", "gear")),
                drs_active=_drs_active(_first(row, "drs", "DRS")),
                source=source,
            )
        )
    return sorted(points, key=lambda item: (item.timestamp, item.driver_code))


def normalize_fastf1_trace_points(
    *,
    rows: list[dict[str, Any]],
    race_id: str,
    session: str,
    source: str = "fastf1",
) -> list[TelemetryTracePoint]:
    """Normalize already-joined FastF1 telemetry rows.

    This helper is mainly for tests and recordings; the existing FastF1 provider
    still streams `TelemetrySample` objects directly.
    """

    points: list[TelemetryTracePoint] = []
    for row in rows or []:
        ts = _parse_datetime(_first(row, "Date", "date", "timestamp", "Time"))
        code = str(_first(row, "Driver", "driver_code", "driver") or "").upper()
        if ts is None or not code:
            continue
        points.append(
            TelemetryTracePoint(
                race_id=race_id,
                session=session,
                driver_code=code,
                driver_number=_safe_int(_first(row, "driver_number", "Number")),
                timestamp=ts,
                lap=_safe_int(_first(row, "LapNumber", "lap_number", "lap")),
                distance_m=_safe_float(_first(row, "Distance", "distance")),
                x=_safe_float(_first(row, "X", "x")),
                y=_safe_float(_first(row, "Y", "y")),
                z=_safe_float(_first(row, "Z", "z")),
                speed_kph=_safe_float(_first(row, "Speed", "speed")),
                throttle_pct=_safe_float(_first(row, "Throttle", "throttle")),
                brake_pct=_brake_pct(_first(row, "Brake", "brake")),
                rpm=_safe_int(_first(row, "RPM", "rpm")),
                gear=_safe_int(_first(row, "nGear", "n_gear", "gear")),
                drs_active=_drs_active(_first(row, "DRS", "drs")),
                source=source,
            )
        )
    return sorted(points, key=lambda item: (item.timestamp, item.driver_code))


def _index_locations(rows: list[dict[str, Any]]) -> dict[int, list[tuple[datetime, dict[str, Any]]]]:
    indexed: dict[int, list[tuple[datetime, dict[str, Any]]]] = {}
    for row in rows:
        number = _safe_int(_first(row, "driver_number", "DriverNumber", "number"))
        ts = _parse_datetime(_first(row, "date", "Date", "timestamp", "Time"))
        if number is None or ts is None:
            continue
        indexed.setdefault(number, []).append((ts, row))
    for values in indexed.values():
        values.sort(key=lambda item: item[0])
    return indexed


def _nearest_location(
    rows: list[tuple[datetime, dict[str, Any]]],
    timestamp: datetime,
    max_gap_seconds: float,
) -> dict[str, Any] | None:
    if not rows:
        return None
    best_ts, best_row = min(rows, key=lambda item: abs((item[0] - timestamp).total_seconds()))
    if abs((best_ts - timestamp).total_seconds()) <= max_gap_seconds:
        return best_row
    return None


def _driver_code(row: dict[str, Any], number: int | None, lookup: dict[int, str]) -> str:
    explicit = _first(row, "driver_code", "Driver", "tla", "Tla")
    if explicit:
        return str(explicit).upper()
    if number is not None and number in lookup:
        return str(lookup[number]).upper()
    return f"#{number}" if number is not None else "UNK"


def _first(row: dict[str, Any] | None, *keys: str) -> Any:
    if not row:
        return None
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _brake_pct(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 100.0 if value else 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric * 100.0 if 0.0 <= numeric <= 1.0 else numeric


def _drs_active(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return int(value) >= 10
    except (TypeError, ValueError):
        return None
