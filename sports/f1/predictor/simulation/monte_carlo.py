"""Lap/session Monte Carlo simulation for F1 predictions."""

from __future__ import annotations

import random
from typing import Any

from sports.f1.predictor.config import MONTE_CARLO_VERSION


class MonteCarloSimulator:
    def __init__(self, iterations: int = 1200, seed: int | None = 42):
        self.iterations = max(1, int(iterations))
        self._random = random.Random(seed)

    def run(self, driver_scores: dict[str, float], context: dict[str, Any] | None = None) -> dict:
        if not driver_scores:
            return {"ok": True, "model_version": MONTE_CARLO_VERSION, "simulation_count": 0, "drivers": []}

        context = context or {}
        laps = max(1, int(context.get("laps") or 57))
        session = str(context.get("session") or "race").lower()
        track = context.get("track") or {}
        weather = context.get("weather") or {}
        tires = context.get("tires") or {}
        reliability = context.get("reliability") or {}
        live_positions = context.get("live_positions") or {}
        live_confidence = max(0.0, min(1.0, float(context.get("live_confidence") or 0.0)))
        live_dynamics = ((context.get("live_dynamics") or {}).get("drivers") or {})

        totals = {
            driver_id: {"wins": 0, "podiums": 0, "top5": 0, "points": 0, "dnfs": 0, "finish_sum": 0.0, "positions": {}}
            for driver_id in driver_scores
        }

        for _ in range(self.iterations):
            classified = []
            for driver_id, score in driver_scores.items():
                driver_reliability = reliability.get(driver_id) or {}
                dnf_probability = float(driver_reliability.get("dnf_probability") or max(0.01, 0.18 - score * 0.12))
                chaos = float(weather.get("chaos_score") or 0.0)
                rain_probability = _num(weather.get("rain_probability"))
                rain_intensity = _num(weather.get("rain_intensity"))
                wind_gusts = _num(weather.get("wind_gusts"), _num(weather.get("wind_speed")))
                temp = _num(weather.get("track_temperature"), _num(weather.get("air_temperature"), 22.0))
                wind_factor = min(1.0, wind_gusts / 70.0)
                cold_factor = max(0.0, min(1.0, (12.0 - temp) / 12.0))
                hot_factor = max(0.0, min(1.0, (temp - 34.0) / 18.0))
                dnf_probability = min(0.58, dnf_probability + chaos * 0.06 + rain_probability * 0.025 + rain_intensity * 0.05 + wind_factor * 0.025)
                if self._random.random() < dnf_probability:
                    totals[driver_id]["dnfs"] += 1
                    pace_total = 999999.0 + self._random.random() * 1000.0
                else:
                    base_lap = _base_lap_time(score, track, session)
                    tire_noise = (float(tires.get("degradation_rate") or track.get("tire_stress") or 0.5) + hot_factor * 0.10) * 0.35
                    weather_noise = chaos * 0.60 + rain_intensity * 0.35 + wind_factor * 0.18 + cold_factor * (0.12 if session == "qualifying" else 0.06)
                    live_anchor = _live_anchor(driver_id, live_positions, live_confidence, live_dynamics)
                    dynamics_noise = _live_dynamics_noise(driver_id, live_dynamics)
                    weekend_variance = self._random.gauss(
                        0.0,
                        1.35
                        + float(track.get("safety_car_probability") or 0.30) * 0.85
                        + chaos * 1.15
                        + rain_probability * 0.75
                        + wind_factor * 0.45,
                    )
                    strategy_variance = self._random.gauss(
                        0.0,
                        0.55
                        + float(tires.get("degradation_rate") or track.get("tire_stress") or 0.50) * 0.85
                        + dynamics_noise * 1.8,
                    )
                    pace_total = weekend_variance + strategy_variance
                    for lap in range(1, laps + 1):
                        lap_age = lap / laps
                        degradation = tire_noise * lap_age
                        safety_variance = float(track.get("safety_car_probability") or 0.30) * self._random.random() * 0.20
                        pace_total += base_lap + degradation + self._random.gauss(0.0, 0.34 + weather_noise + dynamics_noise) + safety_variance
                    pace_total += live_anchor
                classified.append((driver_id, pace_total))

            classified.sort(key=lambda item: item[1])
            for index, (driver_id, _) in enumerate(classified, start=1):
                totals[driver_id]["finish_sum"] += index
                totals[driver_id]["positions"][index] = totals[driver_id]["positions"].get(index, 0) + 1
                if index == 1:
                    totals[driver_id]["wins"] += 1
                if index <= 3:
                    totals[driver_id]["podiums"] += 1
                if index <= 5:
                    totals[driver_id]["top5"] += 1
                if index <= 10:
                    totals[driver_id]["points"] += 1

        drivers = []
        smoothing = 0.75
        smoothed_denominator = self.iterations + smoothing * max(len(totals), 1)
        for driver_id, total in totals.items():
            drivers.append({
                "driver_id": driver_id,
                "win_probability": round((total["wins"] + smoothing) / smoothed_denominator, 4),
                "podium_probability": round((total["podiums"] + smoothing) / smoothed_denominator, 4),
                "top5_probability": round((total["top5"] + smoothing) / smoothed_denominator, 4),
                "points_probability": round((total["points"] + smoothing) / smoothed_denominator, 4),
                "dnf_probability": round(total["dnfs"] / self.iterations, 4),
                "expected_finish": round(total["finish_sum"] / self.iterations, 2),
                "finish_distribution": {
                    str(position): round(count / self.iterations, 4)
                    for position, count in sorted(total["positions"].items())
                },
            })
        drivers.sort(key=lambda row: (row["win_probability"], -row["expected_finish"]), reverse=True)
        return {
            "ok": True,
            "model_version": MONTE_CARLO_VERSION,
            "simulation_count": self.iterations,
            "session": session,
            "laps": laps,
            "drivers": drivers,
        }


def _base_lap_time(score: float, track: dict[str, Any], session: str) -> float:
    length = float(track.get("length_km") or 5.3)
    session_multiplier = 0.985 if session == "qualifying" else 1.0
    # ``score`` is an abstract strength index, not a direct seconds-per-lap
    # estimate. Keeping this coefficient below one second prevents the model
    # from turning small standings/form differences into impossible race locks.
    return (76.0 + length * 6.2 - max(0.0, min(1.0, score)) * 0.92) * session_multiplier


def _live_anchor(
    driver_id: str,
    live_positions: dict[str, Any],
    live_confidence: float = 1.0,
    live_dynamics: dict[str, Any] | None = None,
) -> float:
    dynamic = (live_dynamics or {}).get(driver_id) or {}
    if dynamic.get("time_anchor_seconds") is not None:
        try:
            return float(dynamic.get("time_anchor_seconds"))
        except (TypeError, ValueError):
            pass
    item = live_positions.get(driver_id) or {}
    try:
        position = int(item.get("position"))
        item_confidence = float(item.get("confidence") if item.get("confidence") is not None else live_confidence)
        confidence = max(0.0, min(1.0, item_confidence))
        gap_penalty = _gap_penalty(item)
        return (max(0.0, position - 1) * 1.2 + gap_penalty) * confidence
    except (TypeError, ValueError):
        return 0.0


def _live_dynamics_noise(driver_id: str, live_dynamics: dict[str, Any]) -> float:
    dynamic = live_dynamics.get(driver_id) or {}
    try:
        confidence = max(0.0, min(1.0, float(dynamic.get("confidence") or 0.0)))
        tyre_risk = abs(float(dynamic.get("tyre_delta") or 0.0))
        pit_risk = abs(float(dynamic.get("pit_delta") or 0.0))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, (1.0 - confidence) * 0.05 + tyre_risk * 0.8 + pit_risk * 0.5)


def _gap_penalty(item: dict[str, Any]) -> float:
    value = item.get("gap_to_leader")
    if value in {None, "", "null"}:
        return 0.0
    text = str(value).upper()
    if "LAP" in text:
        return 18.0
    try:
        return min(25.0, max(0.0, float(text.replace("+", ""))) * 0.18)
    except ValueError:
        return 0.0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
