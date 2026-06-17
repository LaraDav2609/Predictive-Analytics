"""Schema for portable F1 ML artifact bundles.

The v1 bundle is deliberately JSON-safe. It can carry direct per-driver learned
inputs today and optional model references later without forcing pickle or a new
runtime into the prediction server.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


ARTIFACT_SCHEMA_VERSION = "f1-ml-artifact-v1"


class F1MLDriverArtifact(BaseModel):
    driver_id: str | None = None
    driver_code: str | None = None
    driver_number: int | None = None
    pace_mean_seconds: float | None = None
    pace_sigma_seconds: float | None = None
    dnf_hazard_per_lap: float | None = None
    rating_prior: float | None = None
    confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    sources: list[str] = Field(default_factory=list)


class F1MLModelArtifact(BaseModel):
    kind: str
    model_ref: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.70, ge=0.0, le=1.0)
    feature_columns: list[str] = Field(default_factory=list)
    source: str | None = None


class F1MLArtifactBundle(BaseModel):
    artifact_id: str
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    model_id: str = "ml_simulator_v1"
    model_version: str = "ml-simulator-artifact-v1"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    training_seasons: list[int] = Field(default_factory=list)
    training_races: list[str] = Field(default_factory=list)
    training_cutoff: datetime | None = None
    knowable_as_of: datetime | None = None
    feature_columns: list[str] = Field(default_factory=list)
    drivers: dict[str, F1MLDriverArtifact] = Field(default_factory=dict)
    pace_model: F1MLModelArtifact | None = None
    dnf_model: F1MLModelArtifact | None = None
    survival_model: F1MLModelArtifact | None = None
    rating_priors: dict[str, float] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    leakage_status: dict[str, Any] = Field(default_factory=lambda: {"status": "unknown"})
    compatibility_version: str = "ml_simulator_v1"
