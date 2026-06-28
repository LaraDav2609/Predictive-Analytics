"""Telemetry analysis primitives for the F1 ML simulator.

The package is intentionally provider-neutral: OpenF1, FastF1, recordings, and
fixtures normalize into the same trace/feature/model contracts before touching
the race simulator.
"""

from sports.f1.ml.telemetry.adapter import apply_telemetry_adjustments
from sports.f1.ml.telemetry.artifacts import load_telemetry_artifact_manifest, score_telemetry_artifact_heads
from sports.f1.ml.telemetry.cache import TelemetryCache
from sports.f1.ml.telemetry.explanations import telemetry_feature_importance
from sports.f1.ml.telemetry.features import build_telemetry_features, build_telemetry_features_from_openf1_session
from sports.f1.ml.telemetry.guard import telemetry_leakage_guard_status
from sports.f1.ml.telemetry.model import build_telemetry_model_output
from sports.f1.ml.telemetry.normalizer import normalize_openf1_trace_points
from sports.f1.ml.telemetry.policy import telemetry_rollout_policy
from sports.f1.ml.telemetry.training import (
    build_telemetry_artifact_manifest,
    export_telemetry_training_rows,
    load_telemetry_label_rows,
    load_telemetry_training_rows,
    load_telemetry_training_payloads,
    save_telemetry_training_rows,
    save_telemetry_artifact_manifest,
)
from sports.f1.ml.telemetry.types import (
    TelemetryFeaturePayload,
    TelemetryFeatureVector,
    TelemetryModelOutput,
    TelemetryTracePoint,
)

__all__ = [
    "TelemetryFeaturePayload",
    "TelemetryFeatureVector",
    "TelemetryModelOutput",
    "TelemetryTracePoint",
    "TelemetryCache",
    "apply_telemetry_adjustments",
    "build_telemetry_features",
    "build_telemetry_features_from_openf1_session",
    "load_telemetry_artifact_manifest",
    "score_telemetry_artifact_heads",
    "build_telemetry_model_output",
    "normalize_openf1_trace_points",
    "telemetry_feature_importance",
    "telemetry_leakage_guard_status",
    "telemetry_rollout_policy",
    "build_telemetry_artifact_manifest",
    "export_telemetry_training_rows",
    "load_telemetry_label_rows",
    "load_telemetry_training_rows",
    "load_telemetry_training_payloads",
    "save_telemetry_training_rows",
    "save_telemetry_artifact_manifest",
]
