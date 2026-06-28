from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class Driver(BaseModel):
    id: str
    number: int | None = None
    code: str
    first_name: str
    last_name: str
    nationality: str
    team: str
    profile_url: str | None = None
    photo_url: str | None = None
    points: float = 0.0
    wins: int = 0
    podiums: int = 0
    position: int | None = None  # championship position


class Constructor(BaseModel):
    id: str
    name: str
    nationality: str
    points: float = 0.0
    wins: int = 0
    position: int | None = None


class RacePrediction(BaseModel):
    driver_predictions: dict[str, "DriverRacePrediction"] = Field(default_factory=dict)
    model_version: str = "f1-points-v1"
    generated_at: datetime | None = None
    data_sources: list[str] = Field(default_factory=list)
    confidence: float | None = None
    model_id: str | None = None
    ml_input_source: str | None = None
    ml_provider_sources: list[str] = Field(default_factory=list)
    ml_fallback_reason: str | None = None
    ml_confidence: float | None = None
    simulator_iterations: int | None = None
    trained_artifacts_used: bool | None = None
    evidence_groups_used: list[str] = Field(default_factory=list)
    ml_artifact_id: str | None = None
    ml_artifact_version: str | None = None
    ml_model_contract_used: bool | None = None
    ml_model_adapters_used: list[str] = Field(default_factory=list)
    ml_model_fallback_reason: str | None = None
    pace_adapter_source: str | None = None
    dnf_adapter_source: str | None = None
    rating_adapter_source: str | None = None
    telemetry_model_used: bool | None = None
    telemetry_model_id: str | None = None
    telemetry_model_version: str | None = None
    telemetry_confidence: float | None = None
    telemetry_source_mode: str | None = None
    telemetry_missing_groups: list[str] = Field(default_factory=list)
    telemetry_fallback_reason: str | None = None
    telemetry_warn_only: bool | None = None
    telemetry_policy: dict = Field(default_factory=dict)
    telemetry_leakage_guard_status: dict = Field(default_factory=dict)
    telemetry_probability_deltas: list[dict] = Field(default_factory=list)


class DriverRacePrediction(BaseModel):
    driver_id: str
    driver_name: str
    win_prob: float
    podium_prob: float
    top5_prob: float
    predicted_position: int | None = None
    form_score: float | None = None
    team_score: float | None = None
    driver_skill_score: float | None = None
    car_performance_score: float | None = None
    performance_score: float | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    sentiment_mentions: int = 0
    personal_news_score: float | None = None
    personal_news_mentions: int = 0
    team_news_score: float | None = None
    team_news_mentions: int = 0
    overall_news_score: float | None = None
    news_win_modifier: float | None = None
    race_sentiment_impact_score: float | None = None
    race_sentiment_delta: float | None = None
    race_sentiment_confidence: float | None = None
    race_sentiment_articles: int = 0
    race_sentiment_explanations: list[str] = Field(default_factory=list)
    wdc_prob: float | None = None
    wdc_modifier: float | None = None
    reliability_score: float | None = None
    dnf_prob: float | None = None
    expected_finish: float | None = None
    qualifying_pace_score: float | None = None
    race_pace_score: float | None = None
    track_fit_score: float | None = None
    tire_strategy_score: float | None = None
    weather_risk_score: float | None = None
    confidence: float | None = None
    recent_summary: str | None = None
    explanation: list[str] = Field(default_factory=list)


class Race(BaseModel):
    round: int
    name: str
    circuit: str
    country: str
    date: datetime
    circuit_id: str | None = None
    race_id: str | None = None  # canonical bridge id "{season}-{round:02d}-{SLUG}"; stamped by the API layer
    locality: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    has_sprint: bool = False
    sessions: list[dict] = Field(default_factory=list)
    status: str = "SCHEDULED"  # SCHEDULED, COMPLETED
    results: list["RaceResult"] = Field(default_factory=list)
    prediction: RacePrediction | None = None


class RaceResult(BaseModel):
    position: int
    driver_id: str
    driver_name: str
    team: str
    time: str | None = None
    points: float = 0.0
    status: str = "Finished"
