"""Lap/session Monte Carlo simulation for F1 predictions."""

from __future__ import annotations

import random
from typing import Any

from f1_predictor.config import MONTE_CARLO_VERSION


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

        totals = {
            driver_id: {"wins": 0, "podiums": 0, "points": 0, "dnfs": 0, "finish_sum": 0.0}
            for driver_id in driver_scores
        }

        for _ in range(self.iterations):
            classified = []
            for driver_id, score in driver_scores.items():
                driver_reliability = reliability.get(driver_id) or {}
                dnf_probability = float(driver_reliability.get("dnf_probability") or max(0.01, 0.18 - score * 0.12))
                chaos = float(weather.get("chaos_score") or 0.0)
                dnf_probability = min(0.55, dnf_probability + chaos * 0.06)
                if self._random.random() < dnf_probability:
                    totals[driver_id]["dnfs"] += 1
                    pace_total = 999999.0 + self._random.random() * 1000.0
                else:
                    base_lap = _base_lap_time(score, track, session)
                    tire_noise = float(tires.get("degradation_rate") or track.get("tire_stress") or 0.5) * 0.35
                    weather_noise = chaos * 0.60
                    live_anchor = _live_anchor(driver_id, live_positions)
                    pace_total = 0.0
                    for lap in range(1, laps + 1):
                        lap_age = lap / laps
                        degradation = tire_noise * lap_age
                        safety_variance = float(track.get("safety_car_probability") or 0.30) * self._random.random() * 0.20
                        pace_total += base_lap + degradation + self._random.gauss(0.0, 0.34 + weather_noise) + safety_variance
                    pace_total += live_anchor
                classified.append((driver_id, pace_total))

            classified.sort(key=lambda item: item[1])
            for index, (driver_id, _) in enumerate(classified, start=1):
                totals[driver_id]["finish_sum"] += index
                if index == 1:
                    totals[driver_id]["wins"] += 1
                if index <= 3:
                    totals[driver_id]["podiums"] += 1
                if index <= 10:
                    totals[driver_id]["points"] += 1

        drivers = []
        for driver_id, total in totals.items():
            drivers.append({
                "driver_id": driver_id,
                "win_probability": round(total["wins"] / self.iterations, 4),
                "podium_probability": round(total["podiums"] / self.iterations, 4),
                "points_probability": round(total["points"] / self.iterations, 4),
                "dnf_probability": round(total["dnfs"] / self.iterations, 4),
                "expected_finish": round(total["finish_sum"] / self.iterations, 2),
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
    return (76.0 + length * 6.2 - max(0.0, min(1.0, score)) * 5.8) * session_multiplier


def _live_anchor(driver_id: str, live_positions: dict[str, Any]) -> float:
    item = live_positions.get(driver_id) or {}
    try:
        position = int(item.get("position"))
        return max(0.0, position - 1) * 1.2
    except (TypeError, ValueError):
        return 0.0
