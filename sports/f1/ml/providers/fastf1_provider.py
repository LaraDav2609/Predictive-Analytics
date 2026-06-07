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
from pathlib import Path

from sports.f1.ml.common.types import Lap, Race, SessionType, TelemetrySample, TireCompound, WeatherFrame
from sports.f1.ml.providers.base import TelemetryProvider

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
                for _, tick in car.iterrows():
                    yield TelemetrySample(
                        race_id=race_id,
                        driver_code=str(lap["Driver"]) if "Driver" in lap else driver_code,
                        timestamp=tick["Date"].to_pydatetime() if hasattr(tick.get("Date"), "to_pydatetime") else tick.get("Date"),
                        lap=lap_n,
                        s_coord=float(tick.get("Distance", 0.0)),
                        speed_kph=float(tick.get("Speed", 0.0)),
                        accel_long_g=0.0,  # FastF1 doesn't expose accel directly; derive in features layer if needed
                        accel_lat_g=0.0,
                        throttle_pct=float(tick["Throttle"]) if tick.get("Throttle") is not None else None,
                        brake_pct=100.0 if bool(tick.get("Brake", False)) else 0.0,
                        rpm=int(tick["RPM"]) if tick.get("RPM") is not None else None,
                        gear=int(tick["nGear"]) if tick.get("nGear") is not None else None,
                        drs_active=bool(tick.get("DRS", 0) >= 10),  # FastF1 DRS code: 10/12/14 = active
                    )

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
