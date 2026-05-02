"""Rolling driver / team form features — recent results, qualifying gap to teammate,
points momentum.

Used as priors for the hierarchical Bayesian strength model and as direct features
for GBM heads. Decay-weighted so the most recent races count more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ewma_per_group(
    df: pd.DataFrame,
    group_col: str,
    sort_cols: list[str],
    value_col: str,
    halflife: float,
    out_col: str,
) -> pd.DataFrame:
    """EWMA per group, sorted on `sort_cols`. Returns one row per
    (group, sort_cols) with the running EWMA up to (and including) that row."""
    if df.empty:
        return pd.DataFrame(columns=[group_col, *sort_cols, out_col])

    df = df.sort_values([group_col, *sort_cols]).copy()
    df[out_col] = (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.ewm(halflife=halflife, adjust=False).mean())
    )
    return df[[group_col, *sort_cols, out_col]].reset_index(drop=True)


def rolling_finish_position(
    results: pd.DataFrame,
    halflife_races: float = 5.0,
    driver_col: str = "driver_code",
    race_col: str = "race",
    position_col: str = "position",
) -> pd.DataFrame:
    """Exponentially-weighted average finish position per driver.

    Output columns: driver_code, race, rolling_finish_position. Lower numbers
    mean stronger recent form.
    """
    if results.empty:
        return pd.DataFrame(columns=[driver_col, race_col, "rolling_finish_position"])
    return _ewma_per_group(
        results,
        group_col=driver_col,
        sort_cols=[race_col],
        value_col=position_col,
        halflife=halflife_races,
        out_col="rolling_finish_position",
    )


def teammate_quali_gap(
    results: pd.DataFrame,
    halflife_races: float = 5.0,
    driver_col: str = "driver_code",
    team_col: str = "team_code",
    race_col: str = "race",
    quali_time_col: str = "quali_time_s",
) -> pd.DataFrame:
    """Exponentially-weighted average qualifying delta to teammate (the cleanest
    car-controlled comparison F1 offers).

    For each (race, team), compute (driver_quali - teammate_quali) where
    teammate is the *other* driver in the same team. Negative = faster than
    teammate. EWMA per driver across races.

    Output columns: driver_code, race, ewma_teammate_quali_gap_s.
    """
    if results.empty:
        return pd.DataFrame(columns=[driver_col, race_col, "ewma_teammate_quali_gap_s"])

    rows = []
    for (race, team), grp in results.groupby([race_col, team_col]):
        # Only paired teammate situations contribute.
        if len(grp) != 2:
            continue
        a, b = grp.iloc[0], grp.iloc[1]
        rows.append({
            driver_col: a[driver_col],
            race_col: race,
            "gap_s": float(a[quali_time_col] - b[quali_time_col]),
        })
        rows.append({
            driver_col: b[driver_col],
            race_col: race,
            "gap_s": float(b[quali_time_col] - a[quali_time_col]),
        })

    if not rows:
        return pd.DataFrame(columns=[driver_col, race_col, "ewma_teammate_quali_gap_s"])

    gaps = pd.DataFrame(rows)
    return _ewma_per_group(
        gaps,
        group_col=driver_col,
        sort_cols=[race_col],
        value_col="gap_s",
        halflife=halflife_races,
        out_col="ewma_teammate_quali_gap_s",
    )


def championship_pressure(
    standings: pd.DataFrame,
    race_round: int,
    season: int,
    driver_col: str = "driver_code",
    points_col: str = "points",
    season_col: str = "season",
    round_col: str = "round",
    n_total_rounds: int = 24,
) -> pd.DataFrame:
    """Heuristic: title-fight drivers race more conservatively late season; back-marker
    drivers chase risk. Encode as scalar in [-1, 1].

    Mapping (season fraction f = race_round / n_total_rounds):
      - For drivers within `pressure_threshold` of the points leader: +f
        (more conservative as season progresses).
      - For drivers > 50 pts behind the leader who are out of contention: -f
        (more risk-tolerant).
      - Mid-table: linear interpolation between the two.

    Output columns: driver_code, championship_pressure ∈ [-1, 1].
    """
    if standings.empty:
        return pd.DataFrame(columns=[driver_col, "championship_pressure"])

    season_round = standings[
        (standings[season_col] == season) & (standings[round_col] == race_round)
    ]
    if season_round.empty:
        return pd.DataFrame(columns=[driver_col, "championship_pressure"])

    leader_points = float(season_round[points_col].max())
    f = max(0.0, min(1.0, race_round / float(max(n_total_rounds, 1))))

    out = []
    for _, row in season_round.iterrows():
        gap = leader_points - float(row[points_col])
        if leader_points <= 0:
            pressure = 0.0
        else:
            # Smooth ramp: gap=0 ⇒ +f, gap=50 ⇒ 0, gap=100+ ⇒ -f
            normalized = max(-1.0, min(1.0, 1.0 - gap / 50.0))
            pressure = float(f * normalized)
        out.append({driver_col: row[driver_col], "championship_pressure": pressure})

    return pd.DataFrame(out)
