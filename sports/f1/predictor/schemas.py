"""Internal schemas for F1 prediction modules.

The public API still returns the existing models from ``sports.f1.models.f1``. These
schemas describe module boundaries inside the predictor package.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    race_id: str | int
    session_stage: str = "race"
    live: bool = False


class FeatureValue(BaseModel):
    value: Any
    confidence: float = 0.0
    source: str = "fallback"
    missing_data: bool = False


class DriverFeatureSet(BaseModel):
    driver_id: str
    team: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    missing_data: list[str] = Field(default_factory=list)


class FeatureSnapshot(BaseModel):
    race_id: str | int | None = None
    session_stage: str = "race"
    generated_at: datetime | None = None
    drivers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    constructors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    track: dict[str, Any] = Field(default_factory=dict)
    weather: dict[str, Any] = Field(default_factory=dict)
    tires: dict[str, Any] = Field(default_factory=dict)
    weekend_evidence: dict[str, Any] = Field(default_factory=dict)
    car_model: dict[str, Any] = Field(default_factory=dict)
    reliability: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sentiment: dict[str, dict[str, Any]] = Field(default_factory=dict)
    missing_data: list[str] = Field(default_factory=list)


class ModelOutput(BaseModel):
    driver_id: str
    score: float
    components: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class DriverPrediction(BaseModel):
    driver_id: str
    driver_name: str
    team: str | None = None
    win_probability: float
    podium_probability: float
    points_probability: float | None = None
    dnf_probability: float | None = None
    expected_finish: float | None = None
    main_factors: list[str] = Field(default_factory=list)
    legacy: dict[str, Any] = Field(default_factory=dict)


class RacePredictionResponse(BaseModel):
    ok: bool = True
    race_id: str | int | None = None
    session_stage: str = "race"
    model_version: str
    simulation_count: int = 0
    predictions: list[DriverPrediction] = Field(default_factory=list)
    model_confidence: float = 0.0
    created_at: datetime | None = None


class SimulationResult(BaseModel):
    ok: bool = True
    session: str
    live: bool = False
    model_version: str
    simulations: list[dict[str, Any]] = Field(default_factory=list)
