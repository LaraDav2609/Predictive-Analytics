"""Outcome distribution → market probability.

Given SimResult, compute P(market) for each tradeable market type:
  - winner            : finish_positions[:, d] == 1
  - podium            : finish_positions[:, d] <= 3
  - top_6 / top_10    : finish_positions[:, d] <= k
  - h2h:A:B           : finish_positions[:, A] < finish_positions[:, B]
  - constructor       : sum over team's two drivers
  - fastest_lap       : fastest_lap_driver == d
  - safety_car        : safety_car_counts >= 1
  - margin_of_victory : winner gap > k seconds
"""

from __future__ import annotations

import numpy as np

from sports.f1.ml.simulator.race_sim import SimResult


def _driver_index(result: SimResult, code: str) -> int:
    try:
        return result.driver_codes.index(code)
    except ValueError as e:
        raise KeyError(f"driver '{code}' not in SimResult.driver_codes") from e


def winner_probabilities(result: SimResult) -> dict[str, float]:
    """P(driver finishes P1) per driver."""
    probs = (result.finish_positions == 1).mean(axis=0)
    return {code: float(p) for code, p in zip(result.driver_codes, probs)}


def podium_probabilities(result: SimResult) -> dict[str, float]:
    """P(driver finishes top 3) per driver."""
    probs = (result.finish_positions <= 3).mean(axis=0)
    return {code: float(p) for code, p in zip(result.driver_codes, probs)}


def top_k_probabilities(result: SimResult, k: int) -> dict[str, float]:
    """P(driver finishes top k) per driver."""
    probs = (result.finish_positions <= k).mean(axis=0)
    return {code: float(p) for code, p in zip(result.driver_codes, probs)}


def h2h_probability(result: SimResult, driver_a: str, driver_b: str) -> float:
    """P(A finishes ahead of B). Includes simulations where one or both DNF
    (DNF positions are sorted to the back, preserving partial order)."""
    a = _driver_index(result, driver_a)
    b = _driver_index(result, driver_b)
    return float((result.finish_positions[:, a] < result.finish_positions[:, b]).mean())


def fastest_lap_probabilities(result: SimResult) -> dict[str, float]:
    """P(driver sets fastest lap of the race)."""
    n_drivers = len(result.driver_codes)
    counts = np.bincount(result.fastest_lap_driver, minlength=n_drivers)
    probs = counts / max(1, len(result.fastest_lap_driver))
    return {code: float(p) for code, p in zip(result.driver_codes, probs)}


def safety_car_probability(result: SimResult, min_count: int = 1) -> float:
    """P(at least `min_count` safety cars in the race)."""
    return float((result.safety_car_counts >= min_count).mean())


def dnf_probabilities(result: SimResult) -> dict[str, float]:
    """P(driver retires from the race) per driver."""
    probs = result.dnf_mask.mean(axis=0)
    return {code: float(p) for code, p in zip(result.driver_codes, probs)}
