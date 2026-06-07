"""Backtest metrics for F1 prediction quality."""

from __future__ import annotations

import math
from typing import Any

from models.f1 import RacePrediction


def evaluate_race(prediction: RacePrediction, actual_results: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = prediction.driver_predictions or {}
    ordered_predictions = sorted(predictions.values(), key=lambda item: item.win_prob, reverse=True)
    probability_by_driver = {driver_id: float(item.win_prob or 0.0) for driver_id, item in predictions.items()}
    actual_winner = next((item.get("driver_id") for item in actual_results if item.get("position") == 1), None)
    actual_podium = [item.get("driver_id") for item in actual_results if isinstance(item.get("position"), int) and item["position"] <= 3]
    actual_points = [item.get("driver_id") for item in actual_results if float(item.get("points") or 0) > 0]
    predicted_winner = ordered_predictions[0].driver_id if ordered_predictions else None
    predicted_top3 = [item.driver_id for item in ordered_predictions[:3]]
    predicted_points = [item.driver_id for item in ordered_predictions[:10]]

    actual_positions = {
        item.get("driver_id"): item.get("position")
        for item in actual_results
        if item.get("driver_id") and isinstance(item.get("position"), int)
    }
    finish_errors = []
    for item in ordered_predictions:
        actual_position = actual_positions.get(item.driver_id)
        if actual_position:
            finish_errors.append(abs(float(item.expected_finish or item.predicted_position or 20) - float(actual_position)))

    brier = _brier_score(probability_by_driver, actual_winner)
    log_loss = -math.log(max(min(probability_by_driver.get(actual_winner or "", 0.0), 0.999999), 0.000001))
    podium_hits = len(set(predicted_top3).intersection(actual_podium))
    points_hits = len(set(predicted_points).intersection(actual_points))

    return {
        "actual_winner": actual_winner,
        "predicted_winner": predicted_winner,
        "winner_hit": predicted_winner == actual_winner,
        "actual_podium": actual_podium,
        "predicted_top3": predicted_top3,
        "podium_hits": podium_hits,
        "podium_hit_rate": round(podium_hits / max(1, min(3, len(actual_podium))), 4),
        "actual_points": actual_points,
        "predicted_points": predicted_points,
        "points_hits": points_hits,
        "points_hit_rate": round(points_hits / max(1, len(actual_points)), 4),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "expected_finish_error": round(sum(finish_errors) / len(finish_errors), 4) if finish_errors else None,
        "winner_probability": round(probability_by_driver.get(actual_winner or "", 0.0), 4),
        "top_probability": round(ordered_predictions[0].win_prob, 4) if ordered_predictions else 0.0,
    }


def summarize_races(race_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [row.get("metrics") or {} for row in race_rows]
    count = len(metrics)
    if count == 0:
        return _empty_summary()
    return {
        "race_count": count,
        "winner_accuracy": round(sum(1 for item in metrics if item.get("winner_hit")) / count, 4),
        "avg_podium_hit_rate": _avg([float(item.get("podium_hit_rate") or 0.0) for item in metrics]),
        "avg_points_hit_rate": _avg([float(item.get("points_hit_rate") or 0.0) for item in metrics]),
        "avg_brier_score": _avg([float(item.get("brier_score") or 0.0) for item in metrics]),
        "avg_log_loss": _avg([float(item.get("log_loss") or 0.0) for item in metrics]),
        "avg_expected_finish_error": _avg([float(item.get("expected_finish_error") or 0.0) for item in metrics if item.get("expected_finish_error") is not None]),
        "calibration_buckets": calibration_buckets(race_rows),
        "top_pick_calibration": top_pick_calibration(race_rows),
        "stage_calibration": stage_calibration(race_rows),
    }


def calibration_buckets(race_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float]] = {}
    for row in race_rows:
        for item in row.get("probability_distribution") or []:
            probability = float(item.get("win_probability") or 0.0)
            bucket_floor = int(probability * 10) * 10
            bucket_key = f"{bucket_floor:02d}-{bucket_floor + 10:02d}%"
            bucket = buckets.setdefault(bucket_key, {"count": 0, "predicted_sum": 0.0, "wins": 0})
            bucket["count"] += 1
            bucket["predicted_sum"] += probability
            if item.get("driver_id") == row.get("actual_winner"):
                bucket["wins"] += 1
    ordered = []
    for key in sorted(buckets):
        bucket = buckets[key]
        count = bucket["count"] or 1
        ordered.append({
            "bucket": key,
            "count": int(bucket["count"]),
            "avg_predicted_probability": round(bucket["predicted_sum"] / count, 4),
            "actual_win_rate": round(bucket["wins"] / count, 4),
        })
    return ordered


def top_pick_calibration(race_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, float]] = {}
    for row in race_rows:
        distribution = sorted(row.get("probability_distribution") or [], key=lambda item: float(item.get("win_probability") or 0.0), reverse=True)
        if not distribution:
            continue
        top = distribution[0]
        probability = float(top.get("win_probability") or 0.0)
        bucket_key = _bucket_key(probability)
        bucket = buckets.setdefault(bucket_key, {"count": 0, "predicted_sum": 0.0, "wins": 0})
        bucket["count"] += 1
        bucket["predicted_sum"] += probability
        if top.get("driver_id") == row.get("actual_winner"):
            bucket["wins"] += 1
    return _bucket_rows(buckets)


def stage_calibration(race_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in race_rows:
        by_stage.setdefault(str(row.get("stage") or "unknown"), []).append(row)
    rows = []
    for stage, items in sorted(by_stage.items()):
        metrics = [item.get("metrics") or {} for item in items]
        rows.append({
            "stage": stage,
            "race_count": len(items),
            "winner_accuracy": round(sum(1 for item in metrics if item.get("winner_hit")) / max(1, len(items)), 4),
            "avg_top_probability": _avg([float((item.get("metrics") or {}).get("top_probability") or 0.0) for item in items]),
            "avg_winner_probability": _avg([float((item.get("metrics") or {}).get("winner_probability") or 0.0) for item in items]),
            "avg_log_loss": _avg([float((item.get("metrics") or {}).get("log_loss") or 0.0) for item in items]),
            "top_pick_calibration": top_pick_calibration(items),
        })
    return rows


def _bucket_key(probability: float) -> str:
    bucket_floor = int(probability * 10) * 10
    return f"{bucket_floor:02d}-{bucket_floor + 10:02d}%"


def _bucket_rows(buckets: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    ordered = []
    for key in sorted(buckets):
        bucket = buckets[key]
        count = bucket["count"] or 1
        ordered.append({
            "bucket": key,
            "count": int(bucket["count"]),
            "avg_predicted_probability": round(bucket["predicted_sum"] / count, 4),
            "actual_win_rate": round(bucket["wins"] / count, 4),
        })
    return ordered


def _brier_score(probability_by_driver: dict[str, float], actual_winner: str | None) -> float:
    if not probability_by_driver:
        return 1.0
    return sum((prob - (1.0 if driver_id == actual_winner else 0.0)) ** 2 for driver_id, prob in probability_by_driver.items()) / len(probability_by_driver)


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _empty_summary() -> dict[str, Any]:
    return {
        "race_count": 0,
        "winner_accuracy": 0.0,
        "avg_podium_hit_rate": 0.0,
        "avg_points_hit_rate": 0.0,
        "avg_brier_score": 0.0,
        "avg_log_loss": 0.0,
        "avg_expected_finish_error": 0.0,
        "calibration_buckets": [],
        "top_pick_calibration": [],
        "stage_calibration": [],
    }
