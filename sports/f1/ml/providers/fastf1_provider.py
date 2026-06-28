"""FastF1 implementation of TelemetryProvider — public broadcast feed.

Coverage: ~10 Hz GPS + speed + lap timing for every race since 2018.
Caching: FastF1's local cache (.fastf1_cache/) — first race fetch downloads
~50 MB and takes 10-30 s; subsequent reads are ~instant.

Tested by `tests/test_fastf1_provider.py` against pure-data conversions.
The integration tests that hit the FastF1 API are marked `@pytest.mark.network`
and skipped by default.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from sports.f1.ml.common.types import Lap, Race, SessionType, TelemetrySample, TireCompound, WeatherFrame
from sports.f1.ml.providers.base import TelemetryProvider
from sports.f1.ml.telemetry.types import TelemetryTracePoint

# Map our SessionType enum to FastF1's session identifier.
_SESSION_MAP: dict[SessionType, str] = {
    SessionType.FP1: "FP1",
    SessionType.FP2: "FP2",
    SessionType.FP3: "FP3",
    SessionType.QUALIFYING: "Q",
    SessionType.SPRINT_QUALI: "SS",
    SessionType.SPRINT: "S",
    SessionType.RACE: "R",
}

_COMPOUND_MAP: dict[str, TireCompound] = {
    "SOFT": TireCompound.SOFT,
    "MEDIUM": TireCompound.MEDIUM,
    "HARD": TireCompound.HARD,
    "INTERMEDIATE": TireCompound.INTERMEDIATE,
    "WET": TireCompound.WET,
}


def _normalize_compound(value: object) -> TireCompound:
    """FastF1 sometimes serves compound as 'SOFT', sometimes lowercase, sometimes None
    on no-tire-data laps. Default to MEDIUM for missing values rather than fail
    — these are typically pre-race or in/out laps that the simulator ignores anyway."""
    if value is None:
        return TireCompound.MEDIUM
    s = str(value).upper().strip()
    return _COMPOUND_MAP.get(s, TireCompound.MEDIUM)


def _race_id(season: int, round_: int, track_code: str) -> str:
    return f"{season}-{round_:02d}-{track_code}"


class FastF1Provider(TelemetryProvider):
    def __init__(self, cache_dir: str = ".fastf1_cache") -> None:
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        # Defer importing fastf1 to first use so the module loads without it.
        self._fastf1 = None

    def _ff1(self):
        if self._fastf1 is None:
            import fastf1
            fastf1.Cache.enable_cache(self.cache_dir)
            self._fastf1 = fastf1
        return self._fastf1

    # ----- TelemetryProvider interface -----

    def list_races(self, season: int) -> list[Race]:
        ff1 = self._ff1()
        schedule = ff1.get_event_schedule(season, include_testing=False)
        races: list[Race] = []
        for _, row in schedule.iterrows():
            country = str(row.get("Country", ""))
            location = str(row.get("Location", ""))
            track_code = (location or country).upper().replace(" ", "_")
            event_date = row.get("EventDate")
            if event_date is None:
                continue
            races.append(Race(
                season=int(season),
                round=int(row["RoundNumber"]),
                track_code=track_code,
                name=str(row.get("EventName", track_code)),
                scheduled_start=event_date.to_pydatetime() if hasattr(event_date, "to_pydatetime") else event_date,
            ))
        return races

    def stream_telemetry(self, race: Race, session: SessionType) -> Iterator[TelemetrySample]:
        ff1 = self._ff1()
        ff1_session = ff1.get_session(race.season, race.round, _SESSION_MAP[session])
        ff1_session.load(telemetry=True, laps=True, weather=False, messages=False)

        race_id = _race_id(race.season, race.round, race.track_code)

        for driver_code in ff1_session.drivers:
            try:
                driver_laps = ff1_session.laps.pick_drivers(driver_code)
            except AttributeError:  # older FastF1 versions
                driver_laps = ff1_session.laps[ff1_session.laps["Driver"] == driver_code]

            for _, lap in driver_laps.iterlaps() if hasattr(driver_laps, "iterlaps") else driver_laps.iterrows():
                try:
                    car = lap.get_car_data().add_distance() if hasattr(lap, "get_car_data") else None
                except Exception:
                    continue
                if car is None or car.empty:
                    continue
                lap_n = int(lap["LapNumber"]) if not isinstance(lap, dict) else int(lap.get("LapNumber", 0))
                previous_tick = None
                for _, tick in car.iterrows():
                    yield TelemetrySample(
                        race_id=race_id,
                        driver_code=str(lap["Driver"]) if "Driver" in lap else driver_code,
                        timestamp=tick["Date"].to_pydatetime() if hasattr(tick.get("Date"), "to_pydatetime") else tick.get("Date"),
                        lap=lap_n,
                        s_coord=float(tick.get("Distance", 0.0)),
                        speed_kph=float(tick.get("Speed", 0.0)),
                        accel_long_g=_derive_longitudinal_accel_g(previous_tick, tick),
                        accel_lat_g=0.0,
                        throttle_pct=float(tick["Throttle"]) if tick.get("Throttle") is not None else None,
                        brake_pct=100.0 if bool(tick.get("Brake", False)) else 0.0,
                        rpm=int(tick["RPM"]) if tick.get("RPM") is not None else None,
                        gear=int(tick["nGear"]) if tick.get("nGear") is not None else None,
                        drs_active=bool(tick.get("DRS", 0) >= 10),  # FastF1 DRS code: 10/12/14 = active
                    )
                    previous_tick = tick

    def trace_points(self, race: Race, session: SessionType) -> list[TelemetryTracePoint]:
        """Return car telemetry joined with position samples where FastF1 exposes them."""
        ff1 = self._ff1()
        ff1_session = ff1.get_session(race.season, race.round, _SESSION_MAP[session])
        ff1_session.load(telemetry=True, laps=True, weather=False, messages=True)

        race_id = _race_id(race.season, race.round, race.track_code)
        points: list[TelemetryTracePoint] = []
        for driver_code in _driver_codes(ff1_session):
            try:
                driver_laps = ff1_session.laps.pick_drivers(driver_code)
            except AttributeError:
                driver_laps = ff1_session.laps[ff1_session.laps["Driver"] == driver_code]

            for _, lap in driver_laps.iterlaps() if hasattr(driver_laps, "iterlaps") else driver_laps.iterrows():
                flags = _lap_quality_flags(lap)
                try:
                    car = lap.get_car_data().add_distance() if hasattr(lap, "get_car_data") else None
                except Exception:
                    car = None
                if car is None or getattr(car, "empty", True):
                    continue
                try:
                    pos = lap.get_pos_data() if hasattr(lap, "get_pos_data") else None
                except Exception:
                    pos = None
                pos_rows = _rows_from_frame(pos)
                lap_n = _int_or_zero(_row_get(lap, "LapNumber"))
                previous_tick = None
                for _, tick in car.iterrows():
                    ts = _timestamp(_row_get(tick, "Date"))
                    if ts is None:
                        continue
                    position = _nearest_position(pos_rows, ts)
                    points.append(
                        TelemetryTracePoint(
                            race_id=race_id,
                            session=session.value,
                            driver_code=str(_row_get(lap, "Driver") or driver_code).upper(),
                            timestamp=ts,
                            source="fastf1",
                            lap=lap_n or None,
                            distance_m=_safe_float(_row_get(tick, "Distance")),
                            x=_safe_float(_row_get(position, "X")) if position else None,
                            y=_safe_float(_row_get(position, "Y")) if position else None,
                            z=_safe_float(_row_get(position, "Z")) if position else None,
                            speed_kph=_safe_float(_row_get(tick, "Speed")),
                            accel_long_g=_derive_longitudinal_accel_g(previous_tick, tick),
                            accel_lat_g=0.0,
                            throttle_pct=_safe_float(_row_get(tick, "Throttle")),
                            brake_pct=100.0 if bool(_row_get(tick, "Brake")) else 0.0,
                            rpm=_safe_int(_row_get(tick, "RPM")),
                            gear=_safe_int(_row_get(tick, "nGear")),
                            drs_active=bool((_safe_int(_row_get(tick, "DRS")) or 0) >= 10),
                            track_status=str(_row_get(lap, "TrackStatus") or "") or None,
                            data_quality_flags=flags + ([] if position else ["position_missing"]),
                        )
                    )
                    previous_tick = tick
        return sorted(points, key=lambda item: (item.timestamp, item.driver_code, item.lap or 0))

    def session_diagnostics(self, race: Race, session: SessionType) -> dict[str, Any]:
        """Expose FastF1 status/message coverage without forcing callers into pandas."""
        ff1 = self._ff1()
        ff1_session = ff1.get_session(race.season, race.round, _SESSION_MAP[session])
        ff1_session.load(telemetry=False, laps=True, weather=False, messages=True)
        track_status = _rows_from_frame(getattr(ff1_session, "track_status", None))
        race_control = _rows_from_frame(getattr(ff1_session, "race_control_messages", None))
        laps = getattr(ff1_session, "laps", None)
        lap_rows = _rows_from_frame(laps)
        return {
            "ok": True,
            "source": "fastf1",
            "session": session.value,
            "track_status": track_status,
            "race_control_messages": race_control,
            "raw_counts": {
                "track_status": len(track_status),
                "race_control_messages": len(race_control),
                "laps": len(lap_rows),
            },
            "data_quality": _session_quality(lap_rows, track_status, race_control),
        }

    def laps(self, race: Race, session: SessionType) -> list[Lap]:
        ff1 = self._ff1()
        ff1_session = ff1.get_session(race.season, race.round, _SESSION_MAP[session])
        ff1_session.load(telemetry=False, laps=True, weather=False, messages=False)

        race_id = _race_id(race.season, race.round, race.track_code)
        laps: list[Lap] = []
        for _, row in ff1_session.laps.iterrows():
            lap_time = row.get("LapTime")
            if lap_time is None or (hasattr(lap_time, "total_seconds") and lap_time.total_seconds() == 0):
                continue
            laps.append(Lap(
                race_id=race_id,
                driver_code=str(row["Driver"]),
                lap_number=int(row["LapNumber"]),
                lap_time_s=float(lap_time.total_seconds()) if hasattr(lap_time, "total_seconds") else float(lap_time),
                sector1_s=_seconds_or_zero(row.get("Sector1Time")),
                sector2_s=_seconds_or_zero(row.get("Sector2Time")),
                sector3_s=_seconds_or_zero(row.get("Sector3Time")),
                compound=_normalize_compound(row.get("Compound")),
                tire_age_laps=_int_or_zero(row.get("TyreLife")),
                position=_int_or_zero(row.get("Position")),
                gap_to_leader_s=None,  # FastF1 doesn't directly expose; derived elsewhere
                gap_ahead_s=None,
                pit_in=row.get("PitInTime") is not None and not _is_nat(row.get("PitInTime")),
                pit_out=row.get("PitOutTime") is not None and not _is_nat(row.get("PitOutTime")),
            ))
        return laps

    def weather(self, race: Race) -> list[WeatherFrame]:
        ff1 = self._ff1()
        ff1_session = ff1.get_session(race.season, race.round, _SESSION_MAP[SessionType.RACE])
        ff1_session.load(telemetry=False, laps=False, weather=True, messages=False)
        race_id = _race_id(race.season, race.round, race.track_code)
        frames: list[WeatherFrame] = []
        for _, row in ff1_session.weather_data.iterrows():
            ts = row.get("Time")
            # weather_data Time is a relative timedelta from session start; combine with session date.
            session_date = ff1_session.date if hasattr(ff1_session, "date") else race.scheduled_start
            timestamp = session_date + ts if hasattr(ts, "total_seconds") else session_date
            frames.append(WeatherFrame(
                race_id=race_id,
                timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
                air_temp_c=float(row.get("AirTemp", 20.0)),
                track_temp_c=float(row.get("TrackTemp", 30.0)),
                humidity_pct=float(row.get("Humidity", 50.0)),
                wind_kph=float(row.get("WindSpeed", 0.0)) * 3.6,  # FastF1 wind is m/s
                rain_intensity=1.0 if bool(row.get("Rainfall", False)) else 0.0,
            ))
        return frames

    def is_live(self) -> bool:
        return False


def _seconds_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _derive_longitudinal_accel_g(previous_tick: Any, tick: Any) -> float:
    if previous_tick is None or tick is None:
        return 0.0
    previous_speed = _safe_float(_row_get(previous_tick, "Speed"))
    speed = _safe_float(_row_get(tick, "Speed"))
    previous_time = _timestamp(_row_get(previous_tick, "Date"))
    timestamp = _timestamp(_row_get(tick, "Date"))
    if previous_speed is None or speed is None or previous_time is None or timestamp is None:
        return 0.0
    elapsed = (timestamp - previous_time).total_seconds()
    if elapsed <= 0:
        return 0.0
    accel_mps2 = ((speed - previous_speed) / 3.6) / elapsed
    return round(max(-6.0, min(6.0, accel_mps2 / 9.80665)), 4)


def _nearest_position(rows: list[dict[str, Any]], timestamp: datetime, max_gap_seconds: float = 1.25) -> dict[str, Any] | None:
    if not rows:
        return None
    usable = [(row, _timestamp(_row_get(row, "Date"))) for row in rows]
    usable = [(row, ts) for row, ts in usable if ts is not None]
    if not usable:
        return None
    row, ts = min(usable, key=lambda item: abs((item[1] - timestamp).total_seconds()))
    return row if abs((ts - timestamp).total_seconds()) <= max_gap_seconds else None


def _rows_from_frame(frame: Any) -> list[dict[str, Any]]:
    if frame is None:
        return []
    if isinstance(frame, list):
        return [dict(row) for row in frame if isinstance(row, dict)]
    if isinstance(frame, tuple):
        return [dict(row) for row in frame if isinstance(row, dict)]
    if hasattr(frame, "iterrows"):
        rows = []
        for _, row in frame.iterrows():
            rows.append({key: _jsonable(value) for key, value in row.items()})
        return rows
    return []


def _driver_codes(ff1_session: Any) -> list[str]:
    codes: list[str] = []
    for value in getattr(ff1_session, "drivers", []) or []:
        code = str(value or "").upper()
        if code:
            codes.append(code)
    if codes:
        return codes
    laps = getattr(ff1_session, "laps", None)
    rows = _rows_from_frame(laps)
    return sorted({str(row.get("Driver") or "").upper() for row in rows if row.get("Driver")})


def _lap_quality_flags(lap: Any) -> list[str]:
    flags: list[str] = []
    if bool(_row_get(lap, "Deleted")):
        flags.append("deleted_lap")
    if _row_get(lap, "PitInTime") is not None and not _is_nat(_row_get(lap, "PitInTime")):
        flags.append("pit_in_lap")
    if _row_get(lap, "PitOutTime") is not None and not _is_nat(_row_get(lap, "PitOutTime")):
        flags.append("pit_out_lap")
    if _is_nat(_row_get(lap, "LapTime")):
        flags.append("lap_time_missing")
    return flags


def _session_quality(lap_rows: list[dict[str, Any]], track_status: list[dict[str, Any]], race_control: list[dict[str, Any]]) -> dict[str, Any]:
    deleted = sum(1 for row in lap_rows if bool(row.get("Deleted")))
    pit_boundary = sum(1 for row in lap_rows if row.get("PitInTime") is not None or row.get("PitOutTime") is not None)
    return {
        "deleted_laps": deleted,
        "pit_boundary_laps": pit_boundary,
        "track_status_available": bool(track_status),
        "race_control_available": bool(race_control),
        "missing_groups": [
            name
            for name, present in (
                ("track_status", bool(track_status)),
                ("race_control_messages", bool(race_control)),
                ("laps", bool(lap_rows)),
            )
            if not present
        ],
    }


def _row_get(row: Any, key: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, "get"):
        try:
            return row.get(key)
        except Exception:
            pass
    try:
        return row[key]
    except Exception:
        return getattr(row, key, None)


def _timestamp(value: Any) -> datetime | None:
    if value is None or _is_nat(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or _is_nat(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or _is_nat(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    if _is_nat(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _is_nat(value: object) -> bool:
    """pandas NaT detection that doesn't crash on non-pandas types."""
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except Exception:
        return False


def _int_or_zero(value: object) -> int:
    """Best-effort int conversion. FastF1 returns NaN for missing fields
    (e.g. Position on a DNF lap, TyreLife on outlaps) — those become 0."""
    if value is None:
        return 0
    try:
        import pandas as pd
        if pd.isna(value):
            return 0
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
