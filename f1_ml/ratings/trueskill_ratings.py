"""TrueSkill — Bayesian skill rating with explicit per-driver uncertainty.

Each driver carries (mu, sigma); after each race, both update via factor-graph
message passing. Naturally handles partial information (DNFs, mixed-quality fields)
and gives confidence intervals out of the box.

Better than ELO for sparse data (~20-22 races/yr). Worse than hierarchical Bayes
because it still doesn't formally separate driver from car.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TrueSkillRating:
    driver_code: str
    mu: float
    sigma: float


def update_after_race(
    pre_race: list[TrueSkillRating],
    finish_order: list[str],
    dnf_drivers: set[str],
) -> list[TrueSkillRating]:
    """Factor-graph update over the partial ordering (finishers strictly ordered;
    DNFs unordered amongst themselves and below all finishers)."""
    raise NotImplementedError("delegate to trueskill library or Microsoft TS factor graph impl")


def conservative_skill(rating: TrueSkillRating, k: float = 3.0) -> float:
    """mu - k*sigma — TrueSkill's standard conservative-estimate formula."""
    return rating.mu - k * rating.sigma
