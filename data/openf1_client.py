"""OpenF1 client for real track traces and live car coordinates."""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from models.f1 import Driver, Race

logger = logging.getLogger(__name__)

OPENF1_BASE_URL = "https://api.openf1.org/v1"
VIEWBOX_WIDTH = 1000
VIEWBOX_HEIGHT = 430
PADDING_X = 70
PADDING_Y = 42
TRACK_GEOMETRY_DIR = Path(__file__).resolve().parent / "track_geometry"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_HEADERS = {
    "User-Agent": "F1PredictorDashboard/1.0 (https://localhost.localdomain)",
}


class OpenF1Client:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=OPENF1_BASE_URL, timeout=35.0)
        self._session_cache: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        self._trace_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._svg_trace_cache: dict[str, dict[str, Any]] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def get_session_data(self, race: Race, session: str = "race", live: bool = False) -> dict[str, Any]:
        session_kind = _session_kind(session)
        session_rows = await self._find_session_candidates(race, session_kind, live)
        if not session_rows:
            return {"ok": False, "source": "openf1", "reason": "openf1_session_unavailable", "session": session_kind}
        row = session_rows[0]
        return {"ok": True, "source": "openf1", "session": session_kind, **row}

    async def get_session_features(self, race: Race, session: str = "race", drivers: list[Driver] | None = None, live: bool = False) -> dict[str, Any]:
        session_info = await self.get_session_data(race, session, live)
        if not session_info.get("ok"):
            return session_info
        session_key = session_info.get("session_key")
        if not session_key:
            return {"ok": False, "source": "openf1", "reason": "openf1_session_key_missing"}

        lap_rows = await self.get_laps(int(session_key))
        position_rows = await self.get_positions(int(session_key))
        interval_rows = await self.get_intervals(int(session_key))
        stint_rows = await self.get_stints(int(session_key))
        pit_rows = await self.get_pits(int(session_key))
        weather_rows = await self.get_weather(int(session_key))
        race_control_rows = await self.get_race_control(int(session_key))
        return {
            "ok": True,
            "source": "openf1",
            "session": session_info.get("session"),
            "session_key": session_key,
            "meeting_key": session_info.get("meeting_key"),
            "laps": _summarize_laps(lap_rows, drivers or []),
            "positions": _summarize_positions(position_rows, drivers or []),
            "intervals": _summarize_intervals(interval_rows, drivers or []),
            "stints": _summarize_stints(stint_rows, drivers or []),
            "pits": _summarize_pits(pit_rows, drivers or []),
            "weather": _summarize_weather(weather_rows),
            "race_control": _summarize_race_control(race_control_rows),
            "raw_counts": {
                "laps": len(lap_rows),
                "positions": len(position_rows),
                "intervals": len(interval_rows),
                "stints": len(stint_rows),
                "pits": len(pit_rows),
                "weather": len(weather_rows),
                "race_control": len(race_control_rows),
            },
        }

    async def get_laps(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/laps", {"session_key": session_key, **filters})

    async def get_positions(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/position", {"session_key": session_key, **filters})

    async def get_intervals(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/intervals", {"session_key": session_key, **filters})

    async def get_stints(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/stints", {"session_key": session_key, **filters})

    async def get_pits(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/pit", {"session_key": session_key, **filters})

    async def get_weather(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/weather", {"session_key": session_key, **filters})

    async def get_race_control(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/race_control", {"session_key": session_key, **filters})

    async def get_car_data(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/car_data", {"session_key": session_key, **filters})

    async def get_location(self, session_key: int, **filters: Any) -> list[dict[str, Any]]:
        return await self._get_list("/location", {"session_key": session_key, **filters})

    async def get_track_data(
        self,
        race: Race,
        drivers: list[Driver],
        session: str = "race",
        live: bool = False,
    ) -> dict[str, Any]:
        """Return normalized OpenF1 trace/live coordinates for an F1 session.

        OpenF1 provides historical location data from 2023 onward and live data
        when the current session is available. If no session/location data exists
        yet, callers receive ok=false and can fall back to local estimates.
        """
        session_kind = _session_kind(session)
        track_key = _track_key(race)
        if track_key in CURATED_PREFERRED_TRACKS and not live:
            return _estimated_trace(race, "preferred_curated_centerline_geometry")

        prefer_svg_geometry = track_key in SVG_PREFERRED_TRACKS and not live
        if prefer_svg_geometry:
            svg_trace = await self._svg_trace(race, "preferred_static_svg_geometry")
            if svg_trace:
                return svg_trace

        session_rows = await self._find_session_candidates(race, session_kind, live)
        if not session_rows:
            if session_kind != "race":
                fallback = await self._fallback_race_trace(race, drivers, f"openf1_{session_kind}_session_unavailable")
                if fallback:
                    return fallback
            svg_trace = await self._svg_trace(race, "openf1_session_unavailable")
            if svg_trace:
                return svg_trace
            return _estimated_trace(race, "openf1_session_unavailable")

        last_failure = _unavailable("openf1_location_trace_unavailable")
        for session_row in session_rows:
            session_key = session_row.get("session_key")
            meeting_key = session_row.get("meeting_key")
            if not session_key:
                last_failure = _unavailable("openf1_session_key_missing")
                continue

            cache_key = (int(session_key), session_kind)
            trace = self._trace_cache.get(cache_key)
            if not trace:
                trace = await self._build_trace(int(session_key), drivers)
                if trace.get("ok"):
                    self._trace_cache[cache_key] = trace

            if not trace.get("ok"):
                last_failure = trace
                continue

            live_positions = await self._latest_locations(
                int(session_key),
                trace.get("bounds") or {},
                drivers,
            ) if live else []

            return _with_track_geometry({
                **trace,
                "integration_version": "openf1-trace-v2",
                "source": "openf1",
                "mode": "live" if live_positions else "historical",
                "meeting_key": meeting_key,
                "session_key": session_key,
                "session_name": session_row.get("session_name"),
                "live_positions": live_positions,
            }, race)

        reason = last_failure.get("reason") or f"openf1_{session_kind}_location_trace_unavailable"
        if session_kind != "race":
            fallback = await self._fallback_race_trace(race, drivers, reason)
            if fallback:
                return fallback
        svg_trace = await self._svg_trace(race, reason)
        if svg_trace:
            return svg_trace
        return _estimated_trace(race, reason)

    async def _svg_trace(self, race: Race, reason: str) -> dict[str, Any] | None:
        key = _track_key(race)
        if key in self._svg_trace_cache:
            cached = dict(self._svg_trace_cache[key])
            cached["reason"] = reason
            return cached
        filename = SVG_TRACK_FILES.get(key)
        if not filename:
            return None
        local_path = TRACK_GEOMETRY_DIR / f"{key}.svg"
        try:
            if local_path.exists():
                svg = local_path.read_text(encoding="utf-8")
                url = local_path.as_posix()
            else:
                url = await self._resolve_wikimedia_svg_url(filename)
                response = await self._client.get(url, follow_redirects=True, timeout=25.0, headers=WIKIMEDIA_HEADERS)
                response.raise_for_status()
                svg = response.text
                try:
                    TRACK_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
                    local_path.write_text(svg, encoding="utf-8")
                except OSError as cache_exc:
                    logger.debug("SVG track disk cache write failed for %s: %s", key, cache_exc)
        except (OSError, httpx.HTTPError, ValueError) as exc:
            logger.warning("SVG track fetch failed for %s: %s", key, exc)
            return None
        trace = _trace_from_svg(svg, key, reason, url)
        if trace.get("ok"):
            self._svg_trace_cache[key] = trace
            return trace
        return None

    async def _resolve_wikimedia_svg_url(self, filename: str) -> str:
        response = await self._client.get(
            WIKIMEDIA_API_URL,
            params={
                "action": "query",
                "titles": f"File:{filename}",
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            },
            timeout=20.0,
            headers=WIKIMEDIA_HEADERS,
        )
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            image_info = page.get("imageinfo") or []
            if image_info and image_info[0].get("url"):
                return str(image_info[0]["url"])
        raise ValueError(f"wikimedia_svg_url_unavailable:{filename}")

    async def _fallback_race_trace(self, race: Race, drivers: list[Driver], reason: str) -> dict[str, Any] | None:
        race_sessions = await self._find_session_candidates(race, "race", False)
        for session_row in race_sessions:
            session_key = session_row.get("session_key")
            meeting_key = session_row.get("meeting_key")
            if not session_key:
                continue
            cache_key = (int(session_key), "race")
            trace = self._trace_cache.get(cache_key)
            if not trace:
                trace = await self._build_trace(int(session_key), drivers)
                if trace.get("ok"):
                    self._trace_cache[cache_key] = trace
            if trace.get("ok"):
                return _with_track_geometry({
                    **trace,
                    "integration_version": "openf1-trace-v2",
                    "source": "openf1",
                    "mode": "historical",
                    "reason": reason,
                    "meeting_key": meeting_key,
                    "session_key": session_key,
                    "session_name": f"{session_row.get('session_name') or 'Race'} trace fallback",
                    "trace_fallback": "race",
                    "live_positions": [],
                }, race)
        return None

    async def _find_session_candidates(self, race: Race, session_kind: str, live: bool) -> list[dict[str, Any]]:
        cache_key = (race.date.year, race.round, session_kind)
        if cache_key in self._session_cache:
            return self._session_cache[cache_key]

        candidates: list[dict[str, Any]] = []
        for year in _candidate_years(race.date.year):
            meetings = await self._get_list("/meetings", {"year": year})
            meeting = _best_meeting_match(meetings, race, year == race.date.year)
            if not meeting:
                continue

            sessions = await self._get_list("/sessions", {"meeting_key": meeting.get("meeting_key")})
            session = _best_session_match(sessions, session_kind)
            if session:
                candidates.append(session)

        self._session_cache[cache_key] = candidates
        return candidates

    async def _build_trace(self, session_key: int, drivers: list[Driver]) -> dict[str, Any]:
        session_drivers = await self._get_list("/drivers", {"session_key": session_key})
        driver_numbers = _trace_driver_numbers(session_drivers, drivers)
        if not driver_numbers:
            return _unavailable("openf1_driver_unavailable")

        raw_points: list[dict[str, float]] = []
        trace_driver_number: int | None = None
        for driver_number in driver_numbers:
            rows = await self._get_list("/location", {
                "session_key": session_key,
                "driver_number": driver_number,
            })
            raw_points = _valid_location_points(rows)
            if len(raw_points) >= 24:
                trace_driver_number = driver_number
                break
        if len(raw_points) < 24 or trace_driver_number is None:
            unavailable = _unavailable("openf1_location_trace_unavailable")
            unavailable["tried_driver_numbers"] = driver_numbers[:8]
            return unavailable

        normalized, bounds = _normalize_points(raw_points)
        simplified = _simplify_points(normalized, max_points=520, min_distance=3.5)
        path = _path_from_points(simplified)
        return _with_track_geometry({
            "ok": True,
            "source": "openf1",
            "mode": "historical",
            "driver_number": trace_driver_number,
            "points": simplified,
            "path": path,
            "bounds": bounds,
            "sample_count": len(raw_points),
        }, race=None)

    async def _latest_locations(
        self,
        session_key: int,
        bounds: dict[str, float],
        drivers: list[Driver],
    ) -> list[dict[str, Any]]:
        since = (datetime.now(timezone.utc) - timedelta(seconds=90)).isoformat()
        rows = await self._get_list("/location", {
            "session_key": session_key,
            "date>": since,
        })
        if not rows:
            return []

        latest_by_driver: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                number = int(row.get("driver_number"))
            except (TypeError, ValueError):
                continue
            current = latest_by_driver.get(number)
            if not current or str(row.get("date") or "") > str(current.get("date") or ""):
                latest_by_driver[number] = row

        driver_by_number = {driver.number: driver for driver in drivers if driver.number is not None}
        positions = []
        for number, row in latest_by_driver.items():
            point = _normalize_point(row, bounds)
            if not point:
                continue
            driver = driver_by_number.get(number)
            positions.append({
                "driver_number": number,
                "driver_id": driver.id if driver else None,
                "driver_code": driver.code if driver else str(number),
                "x": point["x"],
                "y": point["y"],
                "date": row.get("date"),
            })
        return positions

    async def _get_first(self, path: str, params: dict[str, Any]) -> dict[str, Any] | None:
        rows = await self._get_list(path, params)
        return rows[0] if rows else None

    async def _get_list(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            response = await self._client.get(path, params=params)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, list) else []
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("OpenF1 request failed for %s %s: %s", path, params, exc)
            return []


def _driver_code_by_number(drivers: list[Driver]) -> dict[int, str]:
    return {int(driver.number): driver.code for driver in drivers if driver.number is not None}


def _summarize_laps(rows: list[dict[str, Any]], drivers: list[Driver]) -> dict[str, Any]:
    by_number: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            number = int(row.get("driver_number"))
            duration = float(row.get("lap_duration"))
        except (TypeError, ValueError):
            continue
        if duration > 0:
            by_number.setdefault(number, []).append({**row, "lap_duration": duration})
    codes = _driver_code_by_number(drivers)
    driver_rows = {}
    for number, laps in by_number.items():
        durations = [float(row["lap_duration"]) for row in laps]
        clean = sorted(durations)[: max(1, min(5, len(durations)))]
        ordered = sorted(laps, key=lambda item: str(item.get("date") or ""))
        split_at = max(1, len(ordered) // 2)
        early_laps = [float(row["lap_duration"]) for row in ordered[:split_at]]
        late_laps = [float(row["lap_duration"]) for row in ordered[split_at:]] or early_laps
        sector_summary = _sector_summary(laps)
        compounds = sorted({str(row.get("compound")).upper() for row in laps if row.get("compound")})
        long_run_lap = _median(durations) if len(durations) >= 8 else None
        driver_rows[str(number)] = {
            "driver_number": number,
            "driver_code": codes.get(number),
            "laps": len(laps),
            "best_lap": round(min(durations), 3),
            "median_lap": round(_median(durations), 3),
            "representative_lap": round(sum(clean) / len(clean), 3),
            "long_run_lap": round(long_run_lap, 3) if long_run_lap else None,
            "track_evolution_delta": round(min(early_laps) - min(late_laps), 3) if early_laps and late_laps else None,
            "compounds": compounds,
            **sector_summary,
        }
    return {"drivers": driver_rows, "source": "openf1_laps", "missing_data": not bool(driver_rows)}


def _sector_summary(laps: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for index in (1, 2, 3):
        values = []
        for row in laps:
            value = _float_or_none(
                row.get(f"duration_sector_{index}")
                or row.get(f"sector_{index}")
                or row.get(f"sector{index}")
            )
            if value and value > 0:
                values.append(value)
        if not values:
            continue
        clean = sorted(values)[: max(1, min(5, len(values)))]
        payload[f"best_sector_{index}"] = round(min(values), 3)
        payload[f"representative_sector_{index}"] = round(sum(clean) / len(clean), 3)
    payload["sector_coverage"] = round(sum(1 for key in payload if key.startswith("representative_sector_")) / 3.0, 4)
    return payload


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _summarize_positions(rows: list[dict[str, Any]], drivers: list[Driver]) -> dict[str, Any]:
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            number = int(row.get("driver_number"))
            position = int(row.get("position"))
        except (TypeError, ValueError):
            continue
        current = latest.get(number)
        if current is None or str(row.get("date") or "") > str(current.get("date") or ""):
            latest[number] = {**row, "position": position}
    codes = _driver_code_by_number(drivers)
    return {
        "drivers": {
            str(number): {"driver_number": number, "driver_code": codes.get(number), "position": row["position"], "date": row.get("date")}
            for number, row in latest.items()
        },
        "source": "openf1_position",
        "missing_data": not bool(latest),
    }


def _summarize_intervals(rows: list[dict[str, Any]], drivers: list[Driver]) -> dict[str, Any]:
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            number = int(row.get("driver_number"))
        except (TypeError, ValueError):
            continue
        current = latest.get(number)
        if current is None or str(row.get("date") or "") > str(current.get("date") or ""):
            latest[number] = row
    codes = _driver_code_by_number(drivers)
    return {
        "drivers": {
            str(number): {
                "driver_number": number,
                "driver_code": codes.get(number),
                "gap_to_leader": row.get("gap_to_leader"),
                "interval": row.get("interval"),
                "date": row.get("date"),
            }
            for number, row in latest.items()
        },
        "source": "openf1_intervals",
        "missing_data": not bool(latest),
    }


def _summarize_stints(rows: list[dict[str, Any]], drivers: list[Driver]) -> dict[str, Any]:
    by_number: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            number = int(row.get("driver_number"))
        except (TypeError, ValueError):
            continue
        by_number.setdefault(number, []).append(row)
    codes = _driver_code_by_number(drivers)
    return {
        "drivers": {
            str(number): {
                "driver_number": number,
                "driver_code": codes.get(number),
                "stints": len(stints),
                "compounds": sorted({str(item.get("compound")) for item in stints if item.get("compound")}),
                "avg_stint_laps": round(sum(_stint_laps(item) for item in stints) / len(stints), 2) if stints else None,
            }
            for number, stints in by_number.items()
        },
        "source": "openf1_stints",
        "missing_data": not bool(by_number),
    }


def _summarize_pits(rows: list[dict[str, Any]], drivers: list[Driver]) -> dict[str, Any]:
    by_number: dict[int, list[float]] = {}
    for row in rows:
        try:
            number = int(row.get("driver_number"))
            duration = float(row.get("pit_duration"))
        except (TypeError, ValueError):
            continue
        if duration > 0:
            by_number.setdefault(number, []).append(duration)
    codes = _driver_code_by_number(drivers)
    return {
        "drivers": {
            str(number): {
                "driver_number": number,
                "driver_code": codes.get(number),
                "pit_stops": len(durations),
                "avg_pit_duration": round(sum(durations) / len(durations), 3),
            }
            for number, durations in by_number.items()
        },
        "source": "openf1_pit",
        "missing_data": not bool(by_number),
    }


def _summarize_weather(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"source": "openf1_weather", "missing_data": True}
    latest = rows[-1]
    rainfall_rows = [row for row in rows if str(row.get("rainfall")).lower() in {"1", "true", "yes"}]
    return {
        "air_temperature": _float_or_none(latest.get("air_temperature")),
        "track_temperature": _float_or_none(latest.get("track_temperature")),
        "humidity": _float_or_none(latest.get("humidity")),
        "wind_speed": _float_or_none(latest.get("wind_speed")),
        "rain_probability": round(len(rainfall_rows) / len(rows), 4),
        "chaos_score": round(min(1.0, len(rainfall_rows) / len(rows) * 0.75 + (float(latest.get("wind_speed") or 0) / 60.0) * 0.25), 4),
        "samples": len(rows),
        "source": "openf1_weather",
        "missing_data": False,
    }


def _summarize_race_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"source": "openf1_race_control", "missing_data": True}
    text = " ".join(str(row.get("message") or row.get("category") or "").lower() for row in rows)
    safety_count = text.count("safety car") + text.count("virtual safety car")
    flag_count = text.count("yellow") + text.count("red flag")
    return {
        "messages": len(rows),
        "safety_car_messages": safety_count,
        "flag_messages": flag_count,
        "chaos_score": round(min(1.0, safety_count * 0.12 + flag_count * 0.04), 4),
        "source": "openf1_race_control",
        "missing_data": False,
    }


def _stint_laps(row: dict[str, Any]) -> int:
    try:
        start = int(row.get("lap_start"))
        end = int(row.get("lap_end"))
        return max(0, end - start + 1)
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _session_kind(session: str) -> str:
    value = (session or "race").lower()
    normalized = value.replace("_", " ").replace("-", " ")
    if normalized in {"fp1", "free practice 1", "practice 1"} or "practice 1" in normalized:
        return "practice1"
    if normalized in {"fp2", "free practice 2", "practice 2"} or "practice 2" in normalized:
        return "practice2"
    if normalized in {"fp3", "free practice 3", "practice 3"} or "practice 3" in normalized:
        return "practice3"
    if normalized.startswith("fp") or "practice" in normalized:
        return "practice"
    if value.startswith("qual"):
        return "qualifying"
    if value.startswith("sprint"):
        return "sprint"
    return "race"


def _best_session_match(sessions: list[dict[str, Any]], session_kind: str) -> dict[str, Any] | None:
    names = {
        "race": ["race"],
        "qualifying": ["qualifying"],
        "sprint": ["sprint"],
        "practice": ["practice", "fp"],
        "practice1": ["practice 1", "free practice 1", "fp1"],
        "practice2": ["practice 2", "free practice 2", "fp2"],
        "practice3": ["practice 3", "free practice 3", "fp3"],
    }.get(session_kind, ["race"])
    ranked = []
    for item in sessions:
        name = str(item.get("session_name") or "").lower()
        session_type = str(item.get("session_type") or "").lower()
        if any(token in name or token in session_type for token in names):
            ranked.append(item)
    if not ranked and session_kind == "sprint":
        ranked = [item for item in sessions if "sprint" in str(item.get("session_name") or "").lower()]
    if ranked and session_kind == "practice":
        ranked = [item for item in ranked if "practice" in str(item.get("session_name") or "").lower() or str(item.get("session_type") or "").lower() == "practice"]
    return ranked[-1] if ranked else None


def _candidate_years(current_year: int) -> list[int]:
    years = [current_year]
    years.extend(year for year in range(current_year - 1, 2022, -1))
    return years


def _best_meeting_match(meetings: list[dict[str, Any]], race: Race, same_year: bool) -> dict[str, Any] | None:
    if not meetings:
        return None
    track_key = _track_key(race)
    aliases = TRACK_ALIASES.get(track_key, [])
    race_tokens = _tokens(f"{race.name} {race.circuit} {race.country} {' '.join(aliases)}")
    best: tuple[int, dict[str, Any]] | None = None
    for meeting in meetings:
        text = f"{meeting.get('meeting_name', '')} {meeting.get('circuit_short_name', '')} {meeting.get('country_name', '')} {meeting.get('location', '')}"
        normalized_text = _plain(text)
        score = len(race_tokens.intersection(_tokens(text)))
        if aliases and any(alias in normalized_text for alias in aliases):
            score += 30
        try:
            meeting_date = datetime.fromisoformat(str(meeting.get("date_start")).replace("Z", "+00:00"))
            if same_year:
                score += max(0, 12 - abs((meeting_date.date() - race.date.date()).days))
        except (TypeError, ValueError):
            pass
        if best is None or score > best[0]:
            best = (score, meeting)
    return best[1] if best and best[0] > 0 else None


def _tokens(value: str) -> set[str]:
    normalized = _plain(value).replace("-", " ")
    return {part for part in "".join(ch if ch.isalnum() else " " for ch in normalized).split() if len(part) > 2}


def _plain(value: str) -> str:
    replacements = {
        "Ã³": "o", "Ã©": "e", "Ã­": "i", "Ã¡": "a", "Ãº": "u",
        "ó": "o", "é": "e", "í": "i", "á": "a", "ú": "u",
    }
    text = str(value or "").lower()
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


TRACK_ALIASES = {
    "albertpark": ["albert park", "melbourne", "australian", "australia"],
    "shanghai": ["shanghai", "chinese", "china"],
    "suzuka": ["suzuka", "japanese", "japan"],
    "miami": ["miami"],
    "gilles": ["gilles villeneuve", "montreal", "canadian", "canada"],
    "monaco": ["monaco", "monte carlo"],
    "barcelona": ["barcelona", "catalunya", "catalonia"],
    "redbullring": ["red bull ring", "spielberg", "austrian", "austria"],
    "silverstone": ["silverstone", "british", "great britain", "uk"],
    "spa": ["spa-francorchamps", "francorchamps", "belgian", "belgium"],
    "hungaroring": ["hungaroring", "hungarian", "hungary", "budapest"],
    "zandvoort": ["zandvoort", "dutch", "netherlands"],
    "monza": ["monza", "italian", "italy"],
    "madring": ["madring", "madrid"],
    "baku": ["baku", "azerbaijan"],
    "marinabay": ["marina bay", "singapore"],
    "cota": ["cota", "circuit of the americas", "austin", "united states"],
    "mexico": ["mexico", "hermanos rodriguez", "mexico city"],
    "interlagos": ["interlagos", "jose carlos pace", "sao paulo", "brazilian", "brazil"],
    "lasvegas": ["las vegas"],
    "losail": ["losail", "lusail", "qatar"],
    "yasmarina": ["yas marina", "abu dhabi", "uae"],
    "bahrain": ["bahrain", "sakhir"],
    "jeddah": ["jeddah", "saudi arabia"],
    "imola": ["imola", "emilia romagna"],
}


def _track_key(race: Race) -> str:
    text = _plain(f"{race.name} {race.circuit} {race.country}")
    for key, aliases in TRACK_ALIASES.items():
        if any(alias in text for alias in aliases):
            return key
    return "default"


def _trace_driver_numbers(session_drivers: list[dict[str, Any]], drivers: list[Driver]) -> list[int]:
    preferred = [driver.number for driver in drivers if driver.number is not None]
    available = {
        int(row.get("driver_number"))
        for row in session_drivers
        if str(row.get("driver_number") or "").isdigit()
    }
    ordered = [number for number in preferred if number in available]
    ordered.extend(number for number in sorted(available) if number not in ordered)
    ordered.extend(number for number in preferred if number not in ordered)
    return ordered


def _valid_location_points(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    points = []
    for row in rows:
        try:
            x = float(row.get("x"))
            y = float(row.get("y"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append({"x": x, "y": y})
    return points


def _normalize_points(points: list[dict[str, float]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    bounds = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}
    return [_normalize_point(point, bounds) for point in points if _normalize_point(point, bounds)], bounds


def _normalize_point(point: dict[str, Any], bounds: dict[str, float]) -> dict[str, float] | None:
    try:
        x = float(point.get("x"))
        y = float(point.get("y"))
        min_x = float(bounds["min_x"])
        max_x = float(bounds["max_x"])
        min_y = float(bounds["min_y"])
        max_y = float(bounds["max_y"])
    except (KeyError, TypeError, ValueError):
        return None

    width = max(max_x - min_x, 1.0)
    height = max(max_y - min_y, 1.0)
    drawable_w = VIEWBOX_WIDTH - PADDING_X * 2
    drawable_h = VIEWBOX_HEIGHT - PADDING_Y * 2
    scale = min(drawable_w / width, drawable_h / height)
    offset_x = (VIEWBOX_WIDTH - width * scale) / 2
    offset_y = (VIEWBOX_HEIGHT - height * scale) / 2
    return {
        "x": round(offset_x + (x - min_x) * scale, 2),
        "y": round(offset_y + (y - min_y) * scale, 2),
    }


def _simplify_points(points: list[dict[str, float]], max_points: int, min_distance: float) -> list[dict[str, float]]:
    simplified: list[dict[str, float]] = []
    previous: dict[str, float] | None = None
    for point in points:
        if previous is None or math.dist((point["x"], point["y"]), (previous["x"], previous["y"])) >= min_distance:
            simplified.append(point)
            previous = point
    if len(simplified) <= max_points:
        return simplified
    step = len(simplified) / max_points
    return [simplified[min(int(index * step), len(simplified) - 1)] for index in range(max_points)]


def _path_from_points(points: list[dict[str, float]]) -> str:
    if not points:
        return ""
    chunks = [f"M{points[0]['x']} {points[0]['y']}"]
    chunks.extend(f"L{point['x']} {point['y']}" for point in points[1:])
    chunks.append("Z")
    return " ".join(chunks)


def _with_track_geometry(track: dict[str, Any], race: Race | None) -> dict[str, Any]:
    key = _track_key(race) if race else track.get("track_key") or "default"
    source = track.get("source") or "estimated"
    base_points = _coerce_points(track.get("points"))
    if len(base_points) < 3:
        base_points = _points_from_path(str(track.get("path") or ESTIMATED_TRACK_PATHS.get(key) or ESTIMATED_TRACK_PATHS["default"]))
    if len(base_points) < 3:
        base_points = _points_from_path(ESTIMATED_TRACK_PATHS["default"])

    display_points = _simplify_points(base_points, max_points=280, min_distance=2.5)
    racing_line = _resample_closed_line(display_points, samples=360)
    lap_distance = _line_distance(racing_line, closed=True)
    marker_meta = TRACK_GEOMETRY_MARKERS.get(key) or TRACK_GEOMETRY_MARKERS["default"]
    path = track.get("path") or _path_from_points(display_points)
    confidence = _geometry_confidence(source, track, key)

    return {
        **track,
        "ok": True,
        "geometry_version": "f1-track-geometry-v1",
        "track_key": key,
        "path": path,
        "points": track.get("points") or display_points,
        "display_points": display_points,
        "racing_line": racing_line,
        "start_finish_index": int(marker_meta.get("start_finish_index", 0)),
        "lap_distance": round(lap_distance, 2),
        "direction": marker_meta.get("direction", "clockwise"),
        "sectors": marker_meta.get("sectors", []),
        "drs_zones": marker_meta.get("drs_zones", []),
        "pit_entry": marker_meta.get("pit_entry"),
        "pit_exit": marker_meta.get("pit_exit"),
        "markers": _geometry_markers(marker_meta),
        "mapping_source": _mapping_source(source, track),
        "geometry_confidence": confidence,
        "confidence": track.get("confidence", confidence),
    }


def _mapping_source(source: str, track: dict[str, Any]) -> str:
    if source == "openf1" and track.get("sample_count"):
        return "openf1_trace"
    if source == "wikimedia_svg":
        return "wikimedia_svg"
    return "curated_registry"


def point_at_progress(track: dict[str, Any], progress: float, lateral_offset: float = 0.0) -> dict[str, float]:
    line = _coerce_points(track.get("racing_line") or track.get("display_points") or track.get("points"))
    if len(line) < 2:
        line = _points_from_path(str(track.get("path") or ESTIMATED_TRACK_PATHS["default"]))
    if len(line) < 2:
        return {"x": 500.0, "y": 215.0}
    progress = ((float(progress or 0.0) % 1.0) + 1.0) % 1.0
    total = _line_distance(line, closed=True)
    target = total * progress
    walked = 0.0
    for index, start in enumerate(line):
        end = line[(index + 1) % len(line)]
        segment = _distance(start, end)
        if segment <= 0:
            continue
        if walked + segment >= target:
            ratio = (target - walked) / segment
            x = start["x"] + (end["x"] - start["x"]) * ratio
            y = start["y"] + (end["y"] - start["y"]) * ratio
            if lateral_offset:
                nx = -(end["y"] - start["y"]) / segment
                ny = (end["x"] - start["x"]) / segment
                x += nx * lateral_offset
                y += ny * lateral_offset
            return {"x": round(x, 2), "y": round(y, 2)}
        walked += segment
    return {"x": round(line[0]["x"], 2), "y": round(line[0]["y"], 2)}


def progress_from_xy(track: dict[str, Any], x: float, y: float) -> float:
    nearest = nearest_racing_line_point(track, x, y)
    return float(nearest.get("progress", 0.0))


def nearest_racing_line_point(track: dict[str, Any], x: float, y: float) -> dict[str, float]:
    line = _coerce_points(track.get("racing_line") or track.get("display_points") or track.get("points"))
    if len(line) < 2:
        line = _points_from_path(str(track.get("path") or ESTIMATED_TRACK_PATHS["default"]))
    if len(line) < 2:
        return {"x": float(x or 0), "y": float(y or 0), "progress": 0.0, "distance": 0.0}
    total = _line_distance(line, closed=True)
    best: dict[str, float] | None = None
    walked = 0.0
    for index, start in enumerate(line):
        end = line[(index + 1) % len(line)]
        segment = _distance(start, end)
        if segment <= 0:
            continue
        projection = _project_on_segment(float(x), float(y), start, end)
        progress = (walked + segment * projection["t"]) / max(total, 1.0)
        candidate = {"x": projection["x"], "y": projection["y"], "progress": progress % 1.0, "distance": projection["distance"]}
        if best is None or candidate["distance"] < best["distance"]:
            best = candidate
        walked += segment
    return best or {"x": float(x or 0), "y": float(y or 0), "progress": 0.0, "distance": 0.0}


def offset_for_position(position: int, field_size: int = 20) -> float:
    lane = ((max(1, int(position or 1)) - 1) % 5) - 2
    pack = min(1.0, max(0.55, 20 / max(8, int(field_size or 20))))
    return round(lane * 3.2 * pack, 2)


def _geometry_confidence(source: str, track: dict[str, Any], key: str) -> float:
    if source == "openf1" and track.get("sample_count"):
        return 0.92
    if source == "wikimedia_svg":
        return 0.88
    if key in ESTIMATED_TRACK_PATHS:
        return 0.72
    return 0.38


def _geometry_markers(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_finish": meta.get("start_finish", 0.0),
        "sectors": meta.get("sectors", []),
        "drs_zones": meta.get("drs_zones", []),
        "pit_entry": meta.get("pit_entry"),
        "pit_exit": meta.get("pit_exit"),
    }


def _trace_from_svg(svg: str, key: str, reason: str, url: str) -> dict[str, Any]:
    path = _main_course_path(svg)
    if not path:
        return _unavailable("svg_course_path_unavailable")
    try:
        points = _points_from_path(path)
    except ValueError as exc:
        unavailable = _unavailable("svg_course_path_parse_failed")
        unavailable["parse_error"] = str(exc)
        return unavailable
    if len(points) < 24:
        unavailable = _unavailable("svg_course_path_too_sparse")
        unavailable["svg_points"] = len(points)
        return unavailable
    normalized, bounds = _normalize_points(points)
    simplified = _simplify_points(normalized, max_points=640, min_distance=1.0)
    return _with_track_geometry({
        "ok": True,
        "source": "wikimedia_svg",
        "integration_version": "wikimedia-svg-track-v1",
        "mode": "static_vector",
        "reason": reason,
        "track_key": key,
        "points": simplified,
        "path": _path_from_points(simplified),
        "bounds": bounds or _bounds_for_points(points),
        "source_bounds": _bounds_for_points(points),
        "sample_count": len(points),
        "svg_source_url": url,
        "live_positions": [],
    }, None)


def _main_course_path(svg: str) -> str:
    path_matches = re.findall(r"<path\b[^>]*\bd=\"([^\"]+)\"[^>]*>", svg or "", flags=re.IGNORECASE | re.DOTALL)
    if not path_matches:
        return ""
    class_matches = re.findall(r"<path\b(?=[^>]*\bclass=\"st0\")[^>]*\bd=\"([^\"]+)\"[^>]*>", svg or "", flags=re.IGNORECASE | re.DOTALL)
    candidates = class_matches or path_matches
    return max(candidates, key=len).replace("\n", " ").replace("\t", " ").strip()


def _bounds_for_points(points: list[dict[str, float]]) -> dict[str, float]:
    if not points:
        return {}
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}


def _coerce_points(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    points: list[dict[str, float]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append({"x": round(x, 2), "y": round(y, 2)})
    return points


def _points_from_path(path: str) -> list[dict[str, float]]:
    tokens = re.findall(r"[MLHVCSQTAZmlhvcsqtaz]|-?(?:\d+(?:\.\d+)?|\.\d+)", path or "")
    points: list[dict[str, float]] = []
    cursor = {"x": 0.0, "y": 0.0}
    start = {"x": 0.0, "y": 0.0}
    i = 0
    command = ""
    relative = False
    previous_c2: dict[str, float] | None = None
    previous_q1: dict[str, float] | None = None
    command_re = re.compile(r"^[MLHVCSQTAZmlhvcsqtaz]$")
    while i < len(tokens):
        if command_re.match(tokens[i]):
            raw_command = tokens[i]
            relative = raw_command.islower()
            command = raw_command.upper()
            i += 1
        if command == "M" and i + 1 < len(tokens) and not command_re.match(tokens[i]) and not command_re.match(tokens[i + 1]):
            cursor = _path_point(float(tokens[i]), float(tokens[i + 1]), cursor, relative)
            start = dict(cursor)
            points.append(dict(cursor))
            i += 2
            command = "L"
            previous_c2 = None
            previous_q1 = None
        elif command == "L" and i + 1 < len(tokens) and not command_re.match(tokens[i]) and not command_re.match(tokens[i + 1]):
            cursor = _path_point(float(tokens[i]), float(tokens[i + 1]), cursor, relative)
            points.append(dict(cursor))
            i += 2
            previous_c2 = None
            previous_q1 = None
        elif command == "H" and i < len(tokens) and not command_re.match(tokens[i]):
            x = cursor["x"] + float(tokens[i]) if relative else float(tokens[i])
            cursor = {"x": round(x, 2), "y": cursor["y"]}
            points.append(dict(cursor))
            i += 1
            previous_c2 = None
            previous_q1 = None
        elif command == "V" and i < len(tokens) and not command_re.match(tokens[i]):
            y = cursor["y"] + float(tokens[i]) if relative else float(tokens[i])
            cursor = {"x": cursor["x"], "y": round(y, 2)}
            points.append(dict(cursor))
            i += 1
            previous_c2 = None
            previous_q1 = None
        elif command == "C" and i + 5 < len(tokens):
            if any(command_re.match(tokens[i + offset]) for offset in range(6)):
                i += 1
                continue
            p0 = dict(cursor)
            p1 = _path_point(float(tokens[i]), float(tokens[i + 1]), cursor, relative)
            p2 = _path_point(float(tokens[i + 2]), float(tokens[i + 3]), cursor, relative)
            p3 = _path_point(float(tokens[i + 4]), float(tokens[i + 5]), cursor, relative)
            for step in range(1, 15):
                t = step / 14
                points.append(_cubic(p0, p1, p2, p3, t))
            cursor = p3
            previous_c2 = p2
            previous_q1 = None
            i += 6
        elif command == "S" and i + 3 < len(tokens):
            if any(command_re.match(tokens[i + offset]) for offset in range(4)):
                i += 1
                continue
            p0 = dict(cursor)
            p1 = {"x": round(cursor["x"] * 2 - previous_c2["x"], 2), "y": round(cursor["y"] * 2 - previous_c2["y"], 2)} if previous_c2 else dict(cursor)
            p2 = _path_point(float(tokens[i]), float(tokens[i + 1]), cursor, relative)
            p3 = _path_point(float(tokens[i + 2]), float(tokens[i + 3]), cursor, relative)
            for step in range(1, 15):
                t = step / 14
                points.append(_cubic(p0, p1, p2, p3, t))
            cursor = p3
            previous_c2 = p2
            previous_q1 = None
            i += 4
        elif command == "Q" and i + 3 < len(tokens):
            if any(command_re.match(tokens[i + offset]) for offset in range(4)):
                i += 1
                continue
            p0 = dict(cursor)
            p1 = _path_point(float(tokens[i]), float(tokens[i + 1]), cursor, relative)
            p2 = _path_point(float(tokens[i + 2]), float(tokens[i + 3]), cursor, relative)
            for step in range(1, 13):
                t = step / 12
                points.append(_quadratic(p0, p1, p2, t))
            cursor = p2
            previous_q1 = p1
            previous_c2 = None
            i += 4
        elif command == "T" and i + 1 < len(tokens):
            if any(command_re.match(tokens[i + offset]) for offset in range(2)):
                i += 1
                continue
            p0 = dict(cursor)
            p1 = {"x": round(cursor["x"] * 2 - previous_q1["x"], 2), "y": round(cursor["y"] * 2 - previous_q1["y"], 2)} if previous_q1 else dict(cursor)
            p2 = _path_point(float(tokens[i]), float(tokens[i + 1]), cursor, relative)
            for step in range(1, 13):
                t = step / 12
                points.append(_quadratic(p0, p1, p2, t))
            cursor = p2
            previous_q1 = p1
            previous_c2 = None
            i += 2
        elif command == "A" and i + 6 < len(tokens):
            if any(command_re.match(tokens[i + offset]) for offset in range(7)):
                i += 1
                continue
            cursor = _path_point(float(tokens[i + 5]), float(tokens[i + 6]), cursor, relative)
            points.append(dict(cursor))
            previous_c2 = None
            previous_q1 = None
            i += 7
        elif command == "Z":
            if points and _distance(points[-1], start) > 0:
                points.append(dict(start))
            i += 1
            previous_c2 = None
            previous_q1 = None
        else:
            i += 1
    return _dedupe_points(points)


def _path_point(x: float, y: float, cursor: dict[str, float], relative: bool) -> dict[str, float]:
    return {"x": round((cursor["x"] + x) if relative else x, 2), "y": round((cursor["y"] + y) if relative else y, 2)}


def _cubic(p0: dict[str, float], p1: dict[str, float], p2: dict[str, float], p3: dict[str, float], t: float) -> dict[str, float]:
    mt = 1 - t
    return {
        "x": round(mt**3 * p0["x"] + 3 * mt**2 * t * p1["x"] + 3 * mt * t**2 * p2["x"] + t**3 * p3["x"], 2),
        "y": round(mt**3 * p0["y"] + 3 * mt**2 * t * p1["y"] + 3 * mt * t**2 * p2["y"] + t**3 * p3["y"], 2),
    }


def _quadratic(p0: dict[str, float], p1: dict[str, float], p2: dict[str, float], t: float) -> dict[str, float]:
    mt = 1 - t
    return {
        "x": round(mt**2 * p0["x"] + 2 * mt * t * p1["x"] + t**2 * p2["x"], 2),
        "y": round(mt**2 * p0["y"] + 2 * mt * t * p1["y"] + t**2 * p2["y"], 2),
    }


def _dedupe_points(points: list[dict[str, float]]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for point in points:
        if not result or _distance(result[-1], point) > 0.25:
            result.append({"x": round(point["x"], 2), "y": round(point["y"], 2)})
    return result


def _resample_closed_line(points: list[dict[str, float]], samples: int) -> list[dict[str, float]]:
    if len(points) < 2:
        return points
    total = _line_distance(points, closed=True)
    return [point_at_progress({"racing_line": points}, index / samples) for index in range(samples)]


def _line_distance(points: list[dict[str, float]], closed: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    count = len(points) if closed else len(points) - 1
    return sum(_distance(points[index], points[(index + 1) % len(points)]) for index in range(count))


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.dist((float(a["x"]), float(a["y"])), (float(b["x"]), float(b["y"])))


def _project_on_segment(x: float, y: float, start: dict[str, float], end: dict[str, float]) -> dict[str, float]:
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    length_sq = max(dx * dx + dy * dy, 1e-9)
    t = max(0.0, min(1.0, ((x - start["x"]) * dx + (y - start["y"]) * dy) / length_sq))
    px = start["x"] + dx * t
    py = start["y"] + dy * t
    return {"x": round(px, 2), "y": round(py, 2), "t": t, "distance": math.dist((x, y), (px, py))}


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "ok": False,
        "source": "openf1",
        "integration_version": "openf1-trace-v2",
        "mode": "unavailable",
        "reason": reason,
        "points": [],
        "path": "",
        "live_positions": [],
    }


ESTIMATED_TRACK_PATHS = {
    "albertpark": "M168 253 C151 207 180 151 235 125 C318 85 429 108 486 161 C532 204 555 255 629 251 C706 247 776 205 823 239 C879 280 850 350 770 356 C684 362 632 313 562 310 C481 306 454 365 351 363 C256 361 191 319 168 253 Z",
    "shanghai": "M184 253 C156 189 204 121 284 103 C374 84 448 132 443 195 C438 253 365 274 337 229 C315 194 342 161 383 170 C430 181 463 244 532 264 C622 291 688 196 786 203 C868 209 907 283 856 331 C797 387 681 353 591 316 C489 274 421 344 315 348 C240 351 201 313 184 253 Z",
    "suzuka": "M144 244 C201 155 312 111 415 142 C499 168 530 242 618 220 C704 199 737 122 833 147 C911 168 917 256 849 291 C778 327 701 287 628 265 C541 238 487 296 397 326 C280 365 174 333 144 244 C238 205 352 201 442 244 C494 201 552 196 612 222",
    "miami": "M803 111 C714 104 609 101 514 96 C430 92 373 82 320 76 C254 69 222 79 219 97 C216 115 249 130 278 133 C309 137 322 122 350 111 C383 98 421 104 454 122 C501 148 536 174 578 203 C625 236 642 258 617 279 C583 308 511 306 456 289 C399 272 371 239 319 238 C272 237 235 270 192 267 C152 264 130 236 148 203 C166 170 210 160 252 176 C291 191 318 199 350 180 C382 161 414 145 456 147 C511 149 546 177 589 204 C632 231 681 248 730 229 C763 216 775 189 804 178 C828 169 852 183 862 157 C871 132 845 115 803 111 Z",
    "gilles": "M137 257 C173 149 271 112 382 130 C511 151 595 228 724 203 C820 185 884 232 856 300 C826 374 706 372 605 319 C518 274 460 221 355 232 C246 243 176 340 137 257 Z",
    "monaco": "M151 265 C165 194 220 151 289 147 C357 143 390 191 448 181 C505 171 545 115 628 118 C726 121 813 183 808 261 C803 332 726 361 651 329 C588 302 546 271 473 294 C377 324 212 353 151 265 Z",
    "barcelona": "M143 268 C172 145 286 100 410 121 C514 139 545 204 648 200 C760 195 879 221 864 298 C850 371 717 364 610 321 C515 282 460 239 366 268 C262 301 168 341 143 268 Z",
    "redbullring": "M145 261 L240 126 C278 72 364 80 411 136 L507 253 C542 296 606 289 671 249 L785 179 C846 142 912 192 879 257 C842 330 711 352 598 320 L363 253 C275 228 190 330 145 261 Z",
    "silverstone": "M112 238 C165 112 306 102 414 173 C478 214 527 137 645 111 C775 82 901 145 879 250 C858 346 735 334 640 283 C539 228 481 309 353 321 C233 332 137 312 112 238 Z",
    "spa": "M115 289 C145 191 210 119 309 103 C411 87 465 153 521 210 C579 270 660 344 759 326 C840 311 898 239 861 172 C826 110 729 90 634 96 C539 102 509 145 453 116 C358 68 163 129 115 289 Z",
    "hungaroring": "M143 258 C157 157 252 105 360 128 C446 147 467 205 547 178 C637 147 738 154 801 218 C870 288 817 361 706 354 C611 348 561 290 475 303 C365 320 299 366 211 331 C161 311 135 286 143 258 Z",
    "zandvoort": "M151 255 C170 154 263 111 372 121 C466 130 496 197 575 202 C663 208 731 150 809 179 C890 209 899 295 821 335 C739 377 652 319 574 285 C480 245 438 321 333 338 C239 353 134 326 151 255 Z",
    "monza": "M136 267 L248 126 C290 72 388 90 473 132 L604 197 C686 237 748 105 837 136 C918 164 900 293 789 325 C684 355 611 277 506 267 C386 256 250 358 136 267 Z",
    "madring": "M126 273 C153 141 287 86 419 110 C530 130 570 204 650 195 C729 186 745 102 829 127 C915 153 895 279 781 310 C694 334 631 279 560 304 C474 334 422 367 318 346 C221 326 111 340 126 273 Z",
    "baku": "M105 275 L105 144 L311 144 L311 103 L553 103 C718 103 875 139 897 222 C920 309 791 352 628 314 L483 280 L483 328 L232 328 C154 328 105 308 105 275 Z",
    "marinabay": "M127 282 L127 165 L248 165 L248 119 L421 119 L421 170 L538 170 L538 115 L754 115 C838 115 902 170 888 245 C873 326 774 354 683 320 L569 277 L486 328 L305 328 L305 282 Z",
    "cota": "M126 269 C165 126 311 94 445 120 C549 140 593 207 690 201 C786 195 858 152 888 214 C923 289 827 351 714 332 C623 316 578 261 493 278 C384 300 300 365 204 331 C151 312 115 292 126 269 Z",
    "mexico": "M140 267 L140 144 L324 144 C403 144 456 205 530 205 L812 205 C872 205 894 271 850 310 C806 349 720 335 632 302 L507 254 L430 319 L242 319 C173 319 140 301 140 267 Z",
    "interlagos": "M133 253 C183 139 318 101 428 155 C514 197 568 125 690 125 C811 125 884 214 835 291 C788 365 669 349 575 294 C479 238 439 309 341 330 C245 351 160 326 133 253 Z",
    "lasvegas": "M500 386 L812 386 C840 386 854 370 854 343 L854 139 C854 110 833 86 804 85 C781 84 766 99 773 119 C780 139 770 155 747 162 L642 194 C610 204 585 190 568 163 C553 140 530 126 503 126 L451 126 C429 126 414 142 414 164 L414 259 C414 295 386 321 350 322 L285 322 C255 322 235 303 237 276 C239 254 255 239 280 234 L346 221 L346 127 C346 104 329 89 305 89 L133 89 C111 89 95 106 95 129 L95 228 C95 258 79 279 51 289 C33 296 30 318 49 331 L181 382 C211 395 247 400 287 398 L500 386",
    "losail": "M134 260 C161 146 267 101 388 118 C496 133 548 212 650 202 C748 193 836 160 875 221 C922 294 839 361 719 349 C617 339 570 280 484 286 C371 293 291 363 198 330 C151 313 124 291 134 260 Z",
    "yasmarina": "M137 267 C167 150 281 108 408 128 C515 146 549 204 645 194 C747 184 851 161 882 230 C915 305 821 354 703 328 L571 299 L509 342 L294 342 C198 342 119 319 137 267 Z",
    "default": "M105 260 C170 92 365 58 548 84 C690 104 872 82 905 178 C938 274 804 345 646 323 C520 305 471 229 352 253 C241 275 169 337 105 260 Z",
}


SVG_TRACK_FILES = {
    "albertpark": "2022 F1 CourseLayout Australia.svg",
    "shanghai": "Circuit Shanghai.svg",
    "suzuka": "2022 F1 CourseLayout Japan.svg",
    "miami": "2022 F1 CourseLayout Miami.svg",
    "gilles": "2022 F1 CourseLayout Canada.svg",
    "monaco": "2022 F1 CourseLayout Monaco.svg",
    "barcelona": "2022 F1 CourseLayout Spain.svg",
    "redbullring": "2022 F1 CourseLayout Austria.svg",
    "silverstone": "2022 F1 CourseLayout Britain.svg",
    "spa": "2022 F1 CourseLayout Belgium.svg",
    "hungaroring": "2022 F1 CourseLayout Hungary.svg",
    "zandvoort": "2022 F1 CourseLayout Netherlands.svg",
    "monza": "2022 F1 CourseLayout Italia.svg",
    "madring": "Madring (2026).svg",
    "baku": "2022 F1 CourseLayout Azerbaijan.svg",
    "marinabay": "2022 F1 CourseLayout Singapore.svg",
    "cota": "2022 F1 CourseLayout COTA.svg",
    "mexico": "2022 F1 CourseLayout Mexico.svg",
    "interlagos": "2022 F1 CourseLayout São Paulo.svg",
    "lasvegas": "2023 Las Vegas street circuit.svg",
    "losail": "Losail.svg",
    "yasmarina": "2022 F1 CourseLayout Abu Dhabi.svg",
}


SVG_PREFERRED_TRACKS = set(SVG_TRACK_FILES) - {"lasvegas"}
CURATED_PREFERRED_TRACKS = {"lasvegas"}


TRACK_GEOMETRY_MARKERS = {
    key: {
        "start_finish": start,
        "start_finish_index": 0,
        "direction": direction,
        "sectors": [
            {"name": "sector_1", "start": 0.0, "end": 0.333},
            {"name": "sector_2", "start": 0.333, "end": 0.666},
            {"name": "sector_3", "start": 0.666, "end": 1.0},
        ],
        "drs_zones": [{"name": f"DRS {index + 1}", "start": start_at, "end": end_at} for index, (start_at, end_at) in enumerate(drs)],
        "pit_entry": pit[0],
        "pit_exit": pit[1],
    }
    for key, start, direction, drs, pit in [
        ("albertpark", 0.02, "clockwise", [(0.10, 0.18), (0.44, 0.51), (0.73, 0.80), (0.89, 0.96)], (0.92, 0.05)),
        ("shanghai", 0.03, "clockwise", [(0.14, 0.23), (0.67, 0.77)], (0.88, 0.04)),
        ("suzuka", 0.55, "clockwise", [(0.58, 0.67)], (0.48, 0.58)),
        ("miami", 0.01, "clockwise", [(0.10, 0.18), (0.43, 0.51), (0.78, 0.87)], (0.90, 0.04)),
        ("gilles", 0.02, "clockwise", [(0.18, 0.28), (0.62, 0.70), (0.82, 0.94)], (0.91, 0.05)),
        ("monaco", 0.03, "clockwise", [(0.07, 0.15)], (0.78, 0.88)),
        ("barcelona", 0.02, "clockwise", [(0.10, 0.20), (0.72, 0.82)], (0.89, 0.04)),
        ("redbullring", 0.02, "clockwise", [(0.08, 0.20), (0.30, 0.42), (0.74, 0.86)], (0.87, 0.04)),
        ("silverstone", 0.03, "clockwise", [(0.16, 0.27), (0.60, 0.71)], (0.88, 0.04)),
        ("spa", 0.02, "clockwise", [(0.14, 0.25), (0.63, 0.75)], (0.89, 0.04)),
        ("hungaroring", 0.02, "clockwise", [(0.12, 0.21), (0.76, 0.86)], (0.88, 0.04)),
        ("zandvoort", 0.02, "clockwise", [(0.11, 0.20), (0.70, 0.80)], (0.88, 0.04)),
        ("monza", 0.02, "clockwise", [(0.10, 0.22), (0.56, 0.67)], (0.88, 0.04)),
        ("madring", 0.02, "clockwise", [(0.13, 0.23), (0.58, 0.68)], (0.88, 0.04)),
        ("baku", 0.01, "counter_clockwise", [(0.08, 0.20), (0.78, 0.92)], (0.90, 0.04)),
        ("marinabay", 0.02, "counter_clockwise", [(0.11, 0.18), (0.35, 0.43), (0.60, 0.68), (0.79, 0.88)], (0.88, 0.04)),
        ("cota", 0.02, "counter_clockwise", [(0.13, 0.25), (0.62, 0.75)], (0.88, 0.04)),
        ("mexico", 0.02, "clockwise", [(0.11, 0.24), (0.56, 0.66), (0.77, 0.88)], (0.88, 0.04)),
        ("interlagos", 0.02, "counter_clockwise", [(0.13, 0.25), (0.70, 0.84)], (0.88, 0.04)),
        ("lasvegas", 0.01, "counter_clockwise", [(0.14, 0.31), (0.63, 0.78)], (0.88, 0.04)),
        ("losail", 0.02, "clockwise", [(0.11, 0.23)], (0.88, 0.04)),
        ("yasmarina", 0.02, "counter_clockwise", [(0.14, 0.25), (0.62, 0.74)], (0.88, 0.04)),
        ("default", 0.0, "clockwise", [(0.18, 0.30), (0.62, 0.74)], (0.88, 0.04)),
    ]
}


def _estimated_trace(race: Race, reason: str) -> dict[str, Any]:
    key = _track_key(race)
    path = ESTIMATED_TRACK_PATHS.get(key) or ESTIMATED_TRACK_PATHS["default"]
    return _with_track_geometry({
        "ok": True,
        "source": "estimated",
        "integration_version": "estimated-track-v1",
        "mode": "estimated",
        "reason": reason,
        "track_key": key,
        "points": [],
        "path": path,
        "bounds": {},
        "sample_count": 0,
        "live_positions": [],
    }, race)
