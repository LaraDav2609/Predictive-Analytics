"""Core domain dataclasses — provider-agnostic shapes that flow through the pipeline.

Mirrors the .NET F1.Domain entities so cross-process payloads (over Redis) round-trip
cleanly. Pydantic for validation; convert to polars/pandas at feature-extraction time.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TireCompound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class SessionType(str, Enum):
    FP1 = "FP1"
    FP2 = "FP2"
    FP3 = "FP3"
    QUALIFYING = "Q"
    SPRINT_QUALI = "SQ"
    SPRINT = "S"
    RACE = "R"


class Driver(BaseModel):
    code: str  # e.g. "VER", "HAM"
    number: int
    full_name: str
    team_code: str


class Team(BaseModel):
    code: str
    full_name: str


class Race(BaseModel):
    season: int
    round: int
    track_code: str  # e.g. "MONZA", "MONACO"
    name: str
    scheduled_start: datetime


class TelemetrySample(BaseModel):
    """Single 10-25 Hz tick for one car. Provider-normalized."""
    race_id: str
    driver_code: str
    timestamp: datetime
    lap: int
    s_coord: float = Field(description="distance along reference centerline, meters")
    speed_kph: float
    accel_long_g: float
    accel_lat_g: float
    throttle_pct: Optional[float] = None
    brake_pct: Optional[float] = None
    rpm: Optional[int] = None
    gear: Optional[int] = None
    drs_active: Optional[bool] = None


class Lap(BaseModel):
    race_id: str
    driver_code: str
    lap_number: int
    lap_time_s: float
    sector1_s: float
    sector2_s: float
    sector3_s: float
    compound: TireCompound
    tire_age_laps: int
    position: int
    gap_to_leader_s: Optional[float] = None
    gap_ahead_s: Optional[float] = None
    pit_in: bool = False
    pit_out: bool = False


class Stint(BaseModel):
    race_id: str
    driver_code: str
    stint_number: int
    start_lap: int
    end_lap: int
    compound: TireCompound
    laps: int


class PitStop(BaseModel):
    race_id: str
    driver_code: str
    lap: int
    duration_s: float
    new_compound: TireCompound


class WeatherFrame(BaseModel):
    race_id: str
    timestamp: datetime
    air_temp_c: float
    track_temp_c: float
    humidity_pct: float
    wind_kph: float
    rain_intensity: float = Field(ge=0.0, le=1.0)


class RaceOutcomeProbability(BaseModel):
    """Output of the simulator → market mapper. Published to Redis."""
    race_id: str
    driver_code: str
    market: str  # e.g. "winner", "podium", "h2h:VER:HAM"
    probability: float = Field(ge=0.0, le=1.0)
    knowable_as_of: datetime
    model_version: str
