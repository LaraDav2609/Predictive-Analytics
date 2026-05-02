"""Pit stop time distribution per team.

Standard stop is ~2.0-2.5 s stationary, ~22-25 s pit lane delta total. Variance
is the modeling target — Red Bull / McLaren consistently sub-2.5 s; smaller teams
sit near 3 s with heavier left tail (fumbled stops at 5-10+ s).

Model: Gaussian core + heavy-tail mixture (lognormal or shifted-exponential) for
botched stops. Per-team fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PitTimeDistribution:
    team_code: str
    mean_clean_s: float
    sigma_clean_s: float
    botch_probability: float    # P(botched stop)
    botch_mean_extra_s: float   # additional time when botched


def fit(historical_pit_stops: pd.DataFrame) -> dict[str, PitTimeDistribution]:
    """EM mixture: clean Gaussian + heavy-tail component."""
    raise NotImplementedError


def sample(dist: PitTimeDistribution) -> float:
    """One MC draw of the pit-lane delta for this team's next stop."""
    raise NotImplementedError
