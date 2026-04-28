"""Diagnostic calibration recommendations for F1 backtests."""

from __future__ import annotations

from typing import Any


def recommend_weight_adjustments(summary: dict[str, Any]) -> dict[str, Any]:
    winner_accuracy = float(summary.get("winner_accuracy") or 0.0)
    podium_rate = float(summary.get("avg_podium_hit_rate") or 0.0)
    points_rate = float(summary.get("avg_points_hit_rate") or 0.0)
    brier = float(summary.get("avg_brier_score") or 0.0)
    log_loss = float(summary.get("avg_log_loss") or 0.0)
    buckets = summary.get("calibration_buckets") or []
    overconfident = _overconfidence(buckets)

    actions = []
    if winner_accuracy < 0.28:
        actions.append({
            "target": "standing_score",
            "direction": "decrease",
            "magnitude": "small",
            "reason": "Top-pick accuracy is low, so current standings should not dominate pre-race winner forecasts.",
        })
        actions.append({
            "target": "race_pace_score",
            "direction": "increase",
            "magnitude": "medium",
            "reason": "Race outcome selection needs more pace signal relative to standings.",
        })
    if podium_rate < 0.45:
        actions.append({
            "target": "track_fit_score",
            "direction": "increase",
            "magnitude": "small",
            "reason": "Podium hit rate is weak, suggesting circuit-specific suitability needs more influence.",
        })
    if points_rate < 0.60:
        actions.append({
            "target": "reliability_score",
            "direction": "increase",
            "magnitude": "small",
            "reason": "Points prediction is weak, so finish probability and DNF risk should have more weight.",
        })
    if overconfident or log_loss > 2.40:
        actions.append({
            "target": "probability_temperature",
            "direction": "flatten",
            "magnitude": "medium",
            "reason": "Predicted win probabilities are sharper than observed outcomes.",
        })
    if brier < 0.035 and winner_accuracy >= 0.38:
        actions.append({
            "target": "production_weights",
            "direction": "hold",
            "magnitude": "none",
            "reason": "Calibration is acceptable for v1; collect more races before changing weights.",
        })

    return {
        "applies_automatically": False,
        "confidence": _confidence(summary),
        "diagnosis": {
            "overconfident": overconfident,
            "winner_accuracy": winner_accuracy,
            "podium_hit_rate": podium_rate,
            "points_hit_rate": points_rate,
            "brier_score": brier,
            "log_loss": log_loss,
        },
        "actions": actions or [{
            "target": "production_weights",
            "direction": "hold",
            "magnitude": "none",
            "reason": "No strong calibration issue detected in this sample.",
        }],
    }


def _overconfidence(buckets: list[dict[str, Any]]) -> bool:
    tested = [
        bucket for bucket in buckets
        if int(bucket.get("count") or 0) >= 3 and float(bucket.get("avg_predicted_probability") or 0.0) >= 0.20
    ]
    if not tested:
        return False
    return any(
        float(bucket.get("avg_predicted_probability") or 0.0) - float(bucket.get("actual_win_rate") or 0.0) > 0.12
        for bucket in tested
    )


def _confidence(summary: dict[str, Any]) -> float:
    race_count = int(summary.get("race_count") or 0)
    return round(min(0.85, 0.20 + min(1.0, race_count / 40.0) * 0.65), 4)
