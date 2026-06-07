"""Calibration profiles for stage-aware F1 probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalibrationProfile:
    stage: str
    temperature: float
    confidence_scale: float
    confidence_bucket: str
    track_type: str
    adjustments: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "temperature": round(self.temperature, 4),
            "confidence_scale": round(self.confidence_scale, 4),
            "confidence_bucket": self.confidence_bucket,
            "track_type": self.track_type,
            "adjustments": self.adjustments,
        }


BASE_STAGE_PROFILES: dict[str, tuple[float, float]] = {
    "pre_weekend": (1.18, 0.82),
    "practice_available": (1.10, 0.90),
    "post_qualifying": (0.95, 1.00),
    "live": (0.88, 1.08),
    "completed": (0.70, 1.00),
}


def build_calibration_profile(
    stage: str,
    *,
    track: dict[str, Any] | None = None,
    truth: dict[str, Any] | None = None,
    model_temperature: float = 1.0,
    stage_profiles: dict[str, dict[str, float]] | None = None,
) -> CalibrationProfile:
    track = track or {}
    truth = truth or {}
    base_temperature, confidence_scale = BASE_STAGE_PROFILES.get(stage, BASE_STAGE_PROFILES["pre_weekend"])
    if stage_profiles and stage in stage_profiles:
        override = stage_profiles.get(stage) or {}
        base_temperature = float(override.get("temperature") or base_temperature)
        confidence_scale = float(override.get("confidence_scale") or confidence_scale)
    temperature = base_temperature * max(0.70, min(1.40, float(model_temperature or 1.0)))
    adjustments: list[dict[str, Any]] = []

    confidence = max(0.0, min(1.0, float(truth.get("confidence") or 0.0)))
    bucket = _confidence_bucket(confidence)
    if bucket == "low":
        temperature += 0.12
        adjustments.append(_adjustment("confidence", "+0.12", "Low source confidence flattens probability."))
    elif bucket == "high":
        temperature -= 0.04
        adjustments.append(_adjustment("confidence", "-0.04", "High source confidence allows sharper probability."))

    qualifying_importance = _num(track, "qualifying_importance", "quali_importance")
    overtaking_difficulty = _num(track, "overtaking_difficulty", "overtaking")
    tire_stress = _num(track, "tire_stress", "degradation_rate")
    chaos = float(((truth.get("signals") or {}).get("chaos_score") or truth.get("chaos_score") or 0.0))
    track_type = _track_type(track, qualifying_importance, overtaking_difficulty, tire_stress)

    if stage in {"post_qualifying", "live"} and qualifying_importance >= 0.80:
        temperature -= 0.05
        adjustments.append(_adjustment("track_position", "-0.05", "Qualifying-heavy track rewards grid and track position."))
    if stage in {"post_qualifying", "live"} and overtaking_difficulty >= 0.75:
        temperature -= 0.04
        adjustments.append(_adjustment("overtaking", "-0.04", "High overtaking difficulty reduces late-race volatility."))
    if tire_stress >= 0.70:
        temperature += 0.04
        adjustments.append(_adjustment("tires", "+0.04", "High tire stress widens race uncertainty."))
    if chaos > 0.25:
        temperature += 0.06
        adjustments.append(_adjustment("weather_control", "+0.06", "Weather or race-control chaos increases volatility."))

    return CalibrationProfile(
        stage=stage,
        temperature=max(0.55, min(1.60, temperature)),
        confidence_scale=confidence_scale,
        confidence_bucket=bucket,
        track_type=track_type,
        adjustments=adjustments,
    )


def apply_temperature(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    cleaned = {key: max(0.0, float(value or 0.0)) for key, value in probabilities.items()}
    total = sum(cleaned.values())
    if total <= 0:
        uniform = 1.0 / max(len(cleaned), 1)
        return {key: round(uniform, 4) for key in cleaned}
    normalized = {key: value / total for key, value in cleaned.items()}
    exponent = 1.0 / max(0.05, temperature)
    adjusted = {key: value**exponent for key, value in normalized.items()}
    adjusted_total = sum(adjusted.values()) or 1.0
    return {key: round(value / adjusted_total, 4) for key, value in adjusted.items()}


def _confidence_bucket(confidence: float) -> str:
    if confidence < 0.35:
        return "low"
    if confidence > 0.75:
        return "high"
    return "medium"


def _track_type(track: dict[str, Any], qualifying: float, overtaking: float, tires: float) -> str:
    text = f"{track.get('track_type') or ''} {track.get('circuit_type') or ''} {track.get('source') or ''}".lower()
    if "street" in text or qualifying >= 0.82:
        return "street_or_track_position"
    if overtaking <= 0.35:
        return "high_overtaking"
    if tires >= 0.70:
        return "tyre_stress"
    return "balanced"


def _num(source: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = source.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _adjustment(target: str, magnitude: str, reason: str) -> dict[str, str]:
    return {"target": target, "magnitude": magnitude, "reason": reason}
