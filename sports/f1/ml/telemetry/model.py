"""Deterministic V0 telemetry model.

This is the first shippable slice before learned telemetry heads exist. It
turns normalized telemetry features into bounded simulator adjustments with
provenance and confidence.
"""

from __future__ import annotations

from sports.f1.ml.telemetry.artifacts import load_telemetry_artifact_manifest, score_telemetry_artifact_heads
from sports.f1.ml.telemetry.explanations import build_driver_explanation, telemetry_feature_importance
from sports.f1.ml.telemetry.guard import telemetry_leakage_guard_status
from sports.f1.ml.telemetry.types import TelemetryFeaturePayload, TelemetryFeatureVector, TelemetryModelOutput


STAGE_PACE_LIMITS = {
    "pre_weekend": 0.45,
    "practice_available": 0.45,
    "post_qualifying": 0.65,
    "live": 1.20,
    "completed": 1.20,
}

STAGE_CONFIDENCE_SCALE = {
    "pre_weekend": 0.65,
    "practice_available": 0.78,
    "post_qualifying": 0.88,
    "live": 1.00,
    "completed": 1.00,
}


def build_telemetry_model_output(
    payload: TelemetryFeaturePayload,
    *,
    stage: str = "practice_available",
    model_id: str = "telemetry_simulator_v1",
    model_version: str = "telemetry_simulator_v1.v0",
    artifact_path: str | None = None,
) -> TelemetryModelOutput:
    """Build bounded per-driver simulator adjustments from telemetry features."""

    stage = (stage or "practice_available").lower()
    pace_limit = STAGE_PACE_LIMITS.get(stage, 0.45)
    stage_scale = STAGE_CONFIDENCE_SCALE.get(stage, 0.75)
    guard = telemetry_leakage_guard_status(payload, stage=stage, live=bool(payload.data_quality.get("live")))
    artifact_status = load_telemetry_artifact_manifest(artifact_path)
    adjustments: dict[str, dict[str, float]] = {}
    explanations: list[dict] = []
    learned_predictions: dict[str, dict[str, float]] = {}
    for code, feature in payload.driver_features.items():
        learned = score_telemetry_artifact_heads(artifact_status, feature)
        if learned:
            learned_predictions[code] = learned
        raw_pace = learned.get("pace_delta", _raw_pace_shift(feature))
        effective_confidence = min(feature.confidence, payload.confidence) * stage_scale
        pace_shift = _clamp(raw_pace, -pace_limit, pace_limit) * effective_confidence
        pace_sigma_multiplier = _clamp(
            learned.get("pace_quantile", 1.0 + feature.pace_sigma_delta - (feature.stability_score or 0.0) * 0.08),
            0.78,
            1.25,
        )
        tire_delta = _clamp(feature.tire_deg_slope_delta or 0.0, -0.05, 0.09) * effective_confidence
        dnf_multiplier = _clamp(learned.get("dnf_hazard", feature.dnf_hazard_multiplier), 0.75, 1.5)
        overtake_delta = _clamp(learned.get("overtake", ((feature.overtake_pressure or 0.0) - 0.35) * 0.20), -0.07, 0.10) * effective_confidence
        pit_value = _clamp(learned.get("pit_value", _pit_window_value(feature)), -0.25, 0.55) * effective_confidence
        adjustments[code] = {
            "clean_air_pace_shift_s": round(pace_shift, 5),
            "pace_sigma_multiplier": round(pace_sigma_multiplier, 5),
            "tire_deg_slope_delta": round(tire_delta, 5),
            "dnf_hazard_multiplier": round(dnf_multiplier, 5),
            "overtake_score_delta": round(overtake_delta, 5),
            "pit_window_value_s": round(pit_value, 5),
            "confidence": round(effective_confidence, 5),
        }
        explanations.append(build_driver_explanation(code, feature, adjustments[code]))

    return TelemetryModelOutput(
        race_id=payload.race_id,
        session=payload.session,
        model_id=model_id,
        model_version=model_version,
        confidence=payload.confidence,
        source_mode=payload.source_mode,
        driver_adjustments=adjustments,
        top_segments=payload.segment_features,
        probability_delta_explanations=sorted(
            explanations,
            key=lambda row: abs(row.get("pace_shift_s") or 0.0),
            reverse=True,
        )[:8],
        missing_groups=payload.missing_groups,
        metadata={
            "raw_counts": payload.raw_counts,
            "stage": stage,
            "data_quality": payload.data_quality,
            "telemetry_features_version": "telemetry_features_v0",
            "learned_artifacts_used": bool(learned_predictions),
            "learned_artifact_status": artifact_status,
            "learned_head_predictions": learned_predictions,
            "feature_importance": telemetry_feature_importance(),
            "telemetry_leakage_guard_status": guard,
        },
    )


def _raw_pace_shift(feature: TelemetryFeatureVector) -> float:
    shift = 0.60 * feature.clean_air_pace_delta_s
    if feature.top_speed_delta_kph is not None:
        shift -= feature.top_speed_delta_kph * 0.010
    if feature.corner_min_speed_delta_kph is not None:
        shift -= feature.corner_min_speed_delta_kph * 0.015
    if feature.traction_score is not None:
        shift -= (feature.traction_score - 0.50) * 0.18
    if feature.traffic_penalty_s is not None:
        shift += feature.traffic_penalty_s
    if feature.tire_deg_slope_delta is not None:
        shift += feature.tire_deg_slope_delta * 1.5
    return shift


def _pit_window_value(feature: TelemetryFeatureVector) -> float:
    pressure = feature.overtake_pressure or 0.0
    tire = feature.tire_deg_slope_delta or 0.0
    traffic = feature.traffic_penalty_s or 0.0
    return _clamp((pressure * 0.35) + (tire * 1.8) + traffic, -0.25, 0.55)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
