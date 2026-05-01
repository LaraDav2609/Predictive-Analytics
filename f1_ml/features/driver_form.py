"""Rolling driver / team form features — recent results, qualifying gap to teammate,
points momentum.

Used as priors for the hierarchical Bayesian strength model and as direct features
for GBM heads. Decay-weighted so the most recent races count more.
"""

from __future__ import annotations

import pandas as pd


def rolling_finish_position(
    results: pd.DataFrame,
    halflife_races: float = 5.0,
) -> pd.DataFrame:
    """Exponentially-weighted average finish position per driver."""
    raise NotImplementedError


def teammate_quali_gap(
    results: pd.DataFrame,
    halflife_races: float = 5.0,
) -> pd.DataFrame:
    """Exponentially-weighted average qualifying delta to teammate (the cleanest
    car-controlled comparison F1 offers)."""
    raise NotImplementedError


def championship_pressure(
    standings: pd.DataFrame,
    race_round: int,
    season: int,
) -> pd.DataFrame:
    """Heuristic: title-fight drivers race more conservatively late season; back-marker
    drivers chase risk. Encode as scalar in [-1, 1]."""
    raise NotImplementedError
