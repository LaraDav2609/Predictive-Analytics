"""Shared telemetry contracts for provider-normalized F1 analysis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TelemetryTracePoint(BaseModel):
    """A single normalized car/location telemetry point."""

    race_id: str
    session: str
    driver_code: str
    timestamp: datetime
    source: str
    driver_number: int | None = None
    lap: int | None = None
    distance_m: float | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    speed_kph: float | None = None
    accel_long_g: float | None = None
    accel_lat_g: float | None = None
    throttle_pct: float | None = None
    brake_pct: float | None = None
    rpm: int | None = None
    gear: int | None = None
    drs_active: bool | None = None
    track_status: str | None = None
    data_quality_flags: list[str] = Field(default_factory=list)


class TelemetryFeatureVector(BaseModel):
    """Per-driver telemetry features suitable for simulator adjustments."""

    race_id: str
    session: str
    driver_code: str
    source: str
    feature_version: str = "telemetry_features_v0"
    lap: int | None = None
    sector: str | None = None
    clean_air_pace_delta_s: float = 0.0
    pace_sigma_delta: float = 0.0
    full_throttle_pct: float | None = None
    heavy_brake_pct: float | None = None
    corner_min_speed_delta_kph: float | None = None
    top_speed_delta_kph: float | None = None
    traction_score: float | None = None
    stability_score: float | None = None
    tire_deg_slope_delta: float | None = None
    traffic_penalty_s: float | None = None
    overtake_pressure: float | None = None
    dnf_hazard_multiplier: float = 1.0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    samples: int = 0
    raw: dict[str, Any] = Field(default_factory=dict)


class TelemetryFeaturePayload(BaseModel):
    """Feature builder payload returned to model, API, and diagnostics layers."""

    ok: bool = True
    race_id: str
    session: str
    source_mode: str
    confidence: float = Field(ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    driver_features: dict[str, TelemetryFeatureVector] = Field(default_factory=dict)
    lap_features: list[TelemetryFeatureVector] = Field(default_factory=list)
    segment_features: list[dict[str, Any]] = Field(default_factory=list)
    stint_features: list[dict[str, Any]] = Field(default_factory=list)
    raw_counts: dict[str, int] = Field(default_factory=dict)
    missing_groups: list[str] = Field(default_factory=list)
    data_quality: dict[str, Any] = Field(default_factory=dict)


class TelemetryModelOutput(BaseModel):
    """Deterministic or learned telemetry model output before simulator mapping."""

    race_id: str
    session: str
    model_id: str = "telemetry_simulator_v1"
    model_version: str = "telemetry_simulator_v1.v0"
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(ge=0.0, le=1.0)
    source_mode: str
    driver_adjustments: dict[str, dict[str, float]] = Field(default_factory=dict)
    top_segments: list[dict[str, Any]] = Field(default_factory=list)
    probability_delta_explanations: list[dict[str, Any]] = Field(default_factory=list)
    missing_groups: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
