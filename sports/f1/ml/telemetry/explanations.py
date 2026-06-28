"""Explanation helpers for deterministic telemetry model outputs."""

from __future__ import annotations

from typing import Any

from sports.f1.ml.telemetry.types import TelemetryFeatureVector


FEATURE_IMPORTANCE = [
    {
        "feature": "clean_air_pace_delta_s",
        "label": "Clean-air pace",
        "weight": 0.60,
        "direction": "lower_is_faster",
        "model_area": "pace",
        "description": "Primary lap and sector pace signal before simulator mapping.",
    },
    {
        "feature": "traffic_penalty_s",
        "label": "Traffic penalty",
        "weight": 1.00,
        "direction": "higher_is_slower",
        "model_area": "race_state",
        "description": "Adds time when a driver is likely losing pace behind slower cars.",
    },
    {
        "feature": "tire_deg_slope_delta",
        "label": "Tire degradation",
        "weight": 1.50,
        "direction": "higher_is_slower",
        "model_area": "stint",
        "description": "Moves pace and pit-window value when stint fade is visible.",
    },
    {
        "feature": "corner_min_speed_delta_kph",
        "label": "Corner minimum speed",
        "weight": 0.015,
        "direction": "higher_is_faster",
        "model_area": "handling",
        "description": "Rewards low-speed and medium-speed corner strength.",
    },
    {
        "feature": "top_speed_delta_kph",
        "label": "Top speed",
        "weight": 0.010,
        "direction": "higher_is_faster",
        "model_area": "straight_line",
        "description": "Rewards straight-line pace and passability on long straights.",
    },
    {
        "feature": "traction_score",
        "label": "Traction",
        "weight": 0.18,
        "direction": "higher_is_faster",
        "model_area": "handling",
        "description": "Captures throttle application strength after slower corners.",
    },
    {
        "feature": "stability_score",
        "label": "Stability",
        "weight": 0.08,
        "direction": "higher_reduces_uncertainty",
        "model_area": "uncertainty",
        "description": "Reduces pace uncertainty when telemetry is repeatable.",
    },
    {
        "feature": "overtake_pressure",
        "label": "Overtake pressure",
        "weight": 0.20,
        "direction": "higher_improves_attack",
        "model_area": "race_state",
        "description": "Moves overtake score and pit-window value when attack pressure rises.",
    },
    {
        "feature": "dnf_hazard_multiplier",
        "label": "DNF hazard",
        "weight": 1.00,
        "direction": "higher_is_riskier",
        "model_area": "reliability",
        "description": "Scales reliability risk when telemetry anomalies are present.",
    },
]


def telemetry_feature_importance() -> list[dict[str, Any]]:
    """Return the static V0 model importance map for diagnostics."""

    return [dict(item) for item in FEATURE_IMPORTANCE]


def build_driver_explanation(code: str, feature: TelemetryFeatureVector, adjustment: dict[str, float]) -> dict[str, Any]:
    """Build a structured per-driver explanation from features and adjustment."""

    direction = "faster" if adjustment.get("clean_air_pace_shift_s", 0.0) < 0 else "slower"
    factors = _ranked_factors(feature)
    headline = factors[0]["label"] if factors else "Telemetry"
    reason = (
        f"{code} telemetry points to {direction} clean-air pace; "
        f"{headline.lower()} is the leading factor."
    )
    return {
        "driver_code": code,
        "direction": direction,
        "pace_shift_s": adjustment.get("clean_air_pace_shift_s"),
        "confidence": adjustment.get("confidence"),
        "reason": reason,
        "factors": factors[:5],
    }


def _ranked_factors(feature: TelemetryFeatureVector) -> list[dict[str, Any]]:
    factors: list[dict[str, Any]] = []
    _append_factor(factors, "Clean-air pace", "clean_air_pace_delta_s", feature.clean_air_pace_delta_s, abs(feature.clean_air_pace_delta_s) * 0.60, "s")
    _append_factor(factors, "Traffic penalty", "traffic_penalty_s", feature.traffic_penalty_s, abs(feature.traffic_penalty_s or 0.0), "s")
    _append_factor(factors, "Tire degradation", "tire_deg_slope_delta", feature.tire_deg_slope_delta, abs(feature.tire_deg_slope_delta or 0.0) * 1.5, "s/lap")
    _append_factor(factors, "Corner speed", "corner_min_speed_delta_kph", feature.corner_min_speed_delta_kph, abs(feature.corner_min_speed_delta_kph or 0.0) * 0.015, "kph")
    _append_factor(factors, "Top speed", "top_speed_delta_kph", feature.top_speed_delta_kph, abs(feature.top_speed_delta_kph or 0.0) * 0.010, "kph")
    _append_factor(factors, "Traction", "traction_score", feature.traction_score, abs((feature.traction_score or 0.5) - 0.5) * 0.18, "score")
    _append_factor(factors, "Overtake pressure", "overtake_pressure", feature.overtake_pressure, abs((feature.overtake_pressure or 0.35) - 0.35) * 0.20, "score")
    _append_factor(factors, "DNF hazard", "dnf_hazard_multiplier", feature.dnf_hazard_multiplier, abs(feature.dnf_hazard_multiplier - 1.0), "x")
    return sorted(factors, key=lambda row: row["impact"], reverse=True)


def _append_factor(target: list[dict[str, Any]], label: str, key: str, value: float | None, impact: float, unit: str) -> None:
    if value is None:
        return
    target.append({
        "label": label,
        "feature": key,
        "value": round(float(value), 5),
        "impact": round(float(impact), 5),
        "unit": unit,
    })
