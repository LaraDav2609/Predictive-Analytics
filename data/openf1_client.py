"""OpenF1 client for real track traces and live car coordinates."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from models.f1 import Driver, Race

logger = logging.getLogger(__name__)

OPENF1_BASE_URL = "https://api.openf1.org/v1"
VIEWBOX_WIDTH = 1000
VIEWBOX_HEIGHT = 430
PADDING_X = 70
PADDING_Y = 42


class OpenF1Client:
    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=OPENF1_BASE_URL, timeout=35.0)
        self._session_cache: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        self._trace_cache: dict[tuple[int, str], dict[str, Any]] = {}

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
        session_rows = await self._find_session_candidates(race, session_kind, live)
        if not session_rows:
            if session_kind != "race":
                fallback = await self._fallback_race_trace(race, drivers, f"openf1_{session_kind}_session_unavailable")
                if fallback:
                    return fallback
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

            return {
                **trace,
                "integration_version": "openf1-trace-v2",
                "source": "openf1",
                "mode": "live" if live_positions else "historical",
                "meeting_key": meeting_key,
                "session_key": session_key,
                "session_name": session_row.get("session_name"),
                "live_positions": live_positions,
            }

        reason = last_failure.get("reason") or f"openf1_{session_kind}_location_trace_unavailable"
        if session_kind != "race":
            fallback = await self._fallback_race_trace(race, drivers, reason)
            if fallback:
                return fallback
        return _estimated_trace(race, reason)

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
                return {
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
                }
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
        return {
            "ok": True,
            "source": "openf1",
            "mode": "historical",
            "driver_number": trace_driver_number,
            "points": simplified,
            "path": path,
            "bounds": bounds,
            "sample_count": len(raw_points),
        }

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
        driver_rows[str(number)] = {
            "driver_number": number,
            "driver_code": codes.get(number),
            "laps": len(laps),
            "best_lap": round(min(durations), 3),
            "median_lap": round(sorted(durations)[len(durations) // 2], 3),
            "representative_lap": round(sum(clean) / len(clean), 3),
        }
    return {"drivers": driver_rows, "source": "openf1_laps", "missing_data": not bool(driver_rows)}


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
    }[session_kind]
    ranked = []
    for item in sessions:
        name = str(item.get("session_name") or "").lower()
        session_type = str(item.get("session_type") or "").lower()
        if any(token in name or token in session_type for token in names):
            ranked.append(item)
    if not ranked and session_kind == "sprint":
        ranked = [item for item in sessions if "sprint" in str(item.get("session_name") or "").lower()]
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
    "lasvegas": "M111 274 L180 153 L403 153 L461 98 L788 98 C873 98 918 154 886 216 L827 331 L617 331 L549 279 L345 279 L282 331 L169 331 C125 331 94 310 111 274 Z",
    "losail": "M134 260 C161 146 267 101 388 118 C496 133 548 212 650 202 C748 193 836 160 875 221 C922 294 839 361 719 349 C617 339 570 280 484 286 C371 293 291 363 198 330 C151 313 124 291 134 260 Z",
    "yasmarina": "M137 267 C167 150 281 108 408 128 C515 146 549 204 645 194 C747 184 851 161 882 230 C915 305 821 354 703 328 L571 299 L509 342 L294 342 C198 342 119 319 137 267 Z",
    "default": "M105 260 C170 92 365 58 548 84 C690 104 872 82 905 178 C938 274 804 345 646 323 C520 305 471 229 352 253 C241 275 169 337 105 260 Z",
}


def _estimated_trace(race: Race, reason: str) -> dict[str, Any]:
    key = _track_key(race)
    path = ESTIMATED_TRACK_PATHS.get(key) or ESTIMATED_TRACK_PATHS["default"]
    return {
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
    }
