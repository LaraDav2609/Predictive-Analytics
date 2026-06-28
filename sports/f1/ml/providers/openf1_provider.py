"""OpenF1 telemetry bridge for the telemetry simulator model."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sports.f1.ml.telemetry.cache import TelemetryCache
from sports.f1.ml.telemetry.features import build_telemetry_features, build_telemetry_features_from_openf1_session
from sports.f1.ml.telemetry.normalizer import normalize_openf1_trace_points
from sports.f1.ml.telemetry.types import TelemetryFeaturePayload, TelemetryTracePoint


class OpenF1TelemetryProvider:
    """Fetch, cache, and normalize OpenF1 telemetry for model features."""

    def __init__(self, openf1_client: Any, cache: TelemetryCache | None = None) -> None:
        self.openf1 = openf1_client
        self.cache = cache or TelemetryCache()

    async def session_snapshot(
        self,
        *,
        race,
        race_id: str,
        season: int,
        round_num: int,
        session: str,
        drivers: list | None = None,
        live: bool = False,
        include_raw: bool = True,
    ) -> dict[str, Any]:
        session_key = (session or "race").lower()
        if not live:
            cached = self.cache.read_snapshot(
                source="openf1",
                season=season,
                round_num=round_num,
                session=session_key,
                live=False,
            )
            if cached:
                return cached

        openf1_session = await self.openf1.get_session_features(
            race=race,
            session=session_key,
            drivers=drivers or [],
            live=live,
        )
        car_rows: list[dict[str, Any]] = []
        location_rows: list[dict[str, Any]] = []
        if include_raw and openf1_session.get("ok") and openf1_session.get("session_key"):
            filters = _live_filters(live)
            session_id = int(openf1_session["session_key"])
            car_rows = await _call_optional(self.openf1, "get_car_data", session_id, filters)
            location_rows = await _call_optional(self.openf1, "get_location", session_id, filters)

        driver_lookup = _driver_lookup(drivers or [])
        trace_points = normalize_openf1_trace_points(
            race_id=race_id,
            session=session_key,
            driver_lookup=driver_lookup,
            car_rows=car_rows,
            location_rows=location_rows,
            source="openf1_raw",
        )
        raw_counts = dict(openf1_session.get("raw_counts") or {})
        raw_counts.update({
            "car_data": len(car_rows),
            "location": len(location_rows),
            "trace_points": len(trace_points),
        })
        snapshot = {
            "ok": bool(openf1_session.get("ok") or trace_points),
            "source": "openf1",
            "session": session_key,
            "race_id": race_id,
            "openf1_session": openf1_session,
            "car_data": car_rows,
            "location": location_rows,
            "trace_points": [point.model_dump(mode="json") for point in trace_points],
            "raw_counts": raw_counts,
            "last_error": openf1_session.get("last_error"),
        }
        if not snapshot["ok"]:
            snapshot["cache"] = {"hit": False, "persisted": False, "reason": openf1_session.get("reason")}
            return snapshot
        ttl = 8 if live else None
        return self.cache.write_snapshot(
            source="openf1",
            season=season,
            round_num=round_num,
            session=session_key,
            payload=snapshot,
            live=live,
            ttl_seconds=ttl,
        )

    async def telemetry_features(
        self,
        *,
        race,
        race_id: str,
        season: int,
        round_num: int,
        session: str,
        drivers: list | None = None,
        live: bool = False,
    ) -> tuple[TelemetryFeaturePayload, dict[str, Any]]:
        snapshot = await self.session_snapshot(
            race=race,
            race_id=race_id,
            season=season,
            round_num=round_num,
            session=session,
            drivers=drivers,
            live=live,
        )
        points = _trace_points_from_snapshot(snapshot.get("trace_points") or [])
        openf1_session = snapshot.get("openf1_session") or {}
        if points:
            payload = build_telemetry_features(
                race_id=race_id,
                session=(session or "race").lower(),
                trace_points=points,
                laps=_summary_lap_rows(openf1_session),
                stints=_summary_stint_rows(openf1_session),
                intervals=_summary_interval_rows(openf1_session),
                source_mode="openf1_live_raw" if live else "openf1_historical_raw",
                live=live,
            )
            payload.raw_counts.update(snapshot.get("raw_counts") or {})
            payload.data_quality.update({
                "cache_hit": bool((snapshot.get("cache") or {}).get("hit")),
                "raw_trace": True,
                "summary_only": False,
            })
            return payload, snapshot

        payload = build_telemetry_features_from_openf1_session(
            race_id=race_id,
            session=(session or "race").lower(),
            openf1_session=openf1_session,
            live=live,
        )
        payload.data_quality.update({
            "cache_hit": bool((snapshot.get("cache") or {}).get("hit")),
            "raw_trace": False,
            "summary_only": True,
        })
        return payload, snapshot


async def _call_optional(client: Any, method_name: str, session_key: int, filters: dict[str, Any]) -> list[dict[str, Any]]:
    method = getattr(client, method_name, None)
    if method is None:
        return []
    try:
        return await method(session_key, **filters)
    except Exception:
        return []


def _live_filters(live: bool) -> dict[str, Any]:
    if not live:
        return {}
    since = datetime.now(timezone.utc) - timedelta(seconds=90)
    return {"date>=": since.isoformat().replace("+00:00", "Z")}


def _driver_lookup(drivers: list) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for driver in drivers or []:
        number = _safe_int(getattr(driver, "permanent_number", None) or getattr(driver, "number", None))
        code = getattr(driver, "code", None)
        if number is not None and code:
            lookup[number] = str(code).upper()
    return lookup


def _trace_points_from_snapshot(rows: list[dict[str, Any]]) -> list[TelemetryTracePoint]:
    points: list[TelemetryTracePoint] = []
    for row in rows or []:
        try:
            points.append(TelemetryTracePoint.model_validate(row))
        except Exception:
            continue
    return points


def _summary_lap_rows(openf1_session: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in ((openf1_session.get("laps") or {}).get("drivers") or {}).values():
        if not isinstance(row, dict):
            continue
        rows.append({
            "driver_code": row.get("driver_code"),
            "lap_time_s": row.get("representative_lap") or row.get("median_lap") or row.get("best_lap"),
        })
    return rows


def _summary_stint_rows(openf1_session: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"driver_code": row.get("driver_code"), **row}
        for row in ((openf1_session.get("stints") or {}).get("drivers") or {}).values()
        if isinstance(row, dict)
    ]


def _summary_interval_rows(openf1_session: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"driver_code": row.get("driver_code"), **row}
        for row in ((openf1_session.get("intervals") or {}).get("drivers") or {}).values()
        if isinstance(row, dict)
    ]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
