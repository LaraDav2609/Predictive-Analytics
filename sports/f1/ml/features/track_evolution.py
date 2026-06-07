"""Track evolution / grip-up model — track surface gets faster Friday → Sunday
as rubber is laid down, and within each session as more laps accumulate.

Without this, FP3 pace is overestimated (it's faster than it would be in the race
on the same conditions because of grip evolution).

Fit a multiplicative grip factor per session that brings observed pace to a
common (race) reference. Apply inverse when projecting practice pace onto race pace.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_SESSIONS = ("FP1", "FP2", "FP3", "Q", "R")
_DEFAULT_FACTORS = {
    # Sensible defaults if a session has no history (FP1 is a touch slower than
    # race, FP3 close to quali, race typically 1-3% slower than quali pace).
    "FP1": 1.015,
    "FP2": 1.008,
    "FP3": 1.005,
    "Q": 0.985,
    "R": 1.0,
}


@dataclass
class EvolutionFit:
    track_code: str
    fp1_factor: float
    fp2_factor: float
    fp3_factor: float
    quali_factor: float
    race_factor: float = 1.0  # reference

    def factor_for(self, session: str) -> float:
        s = session.upper()
        if s == "FP1":
            return self.fp1_factor
        if s == "FP2":
            return self.fp2_factor
        if s == "FP3":
            return self.fp3_factor
        if s in ("Q", "QUALIFYING", "QUALI"):
            return self.quali_factor
        if s in ("R", "RACE"):
            return self.race_factor
        raise ValueError(f"unknown session '{session}'")


def fit(
    historical_session_pace: pd.DataFrame,
    track_col: str = "track_code",
    session_col: str = "session",
    pace_col: str = "median_pace_s",
) -> EvolutionFit:
    """Estimate session-to-session grip multipliers.

    Expected schema: one row per (track, session, race) with the median
    representative pace (in seconds) for that session. Factor is computed as
    (median session pace) / (median race pace) at the same track. Returned
    factor > 1 means the session is *slower* than race; < 1 means faster.
    """
    if historical_session_pace.empty:
        raise ValueError("historical_session_pace is empty")

    # Single-track scope: callers fit per-track. If multiple are present, take
    # the most-common one and warn implicitly via the returned track_code.
    track = historical_session_pace[track_col].mode().iloc[0]
    df = historical_session_pace[historical_session_pace[track_col] == track]

    medians: dict[str, float] = {}
    for session, grp in df.groupby(session_col):
        s = str(session).upper()
        medians[s] = float(grp[pace_col].median())

    race_baseline = medians.get("R") or medians.get("RACE")
    if race_baseline is None:
        raise ValueError(
            "cannot fit EvolutionFit without race-session pace samples (session='R')"
        )

    factors: dict[str, float] = {"R": 1.0}
    for s in _SESSIONS:
        if s == "R":
            continue
        if s in medians and medians[s] > 0:
            factors[s] = medians[s] / race_baseline
        else:
            factors[s] = _DEFAULT_FACTORS[s]

    return EvolutionFit(
        track_code=str(track),
        fp1_factor=factors["FP1"],
        fp2_factor=factors["FP2"],
        fp3_factor=factors["FP3"],
        quali_factor=factors["Q"],
        race_factor=1.0,
    )


def normalize_to_race(pace_s: float, session: str, fit_: EvolutionFit) -> float:
    """Convert a practice / quali pace estimate to its race-equivalent.

    If FP3 typically runs 1.005 × race pace at this track, an FP3 lap of
    80.5 s implies a race-equivalent of 80.5 / 1.005 = 80.10 s.
    """
    factor = fit_.factor_for(session)
    if factor <= 0:
        raise ValueError(f"invalid factor {factor} for session '{session}'")
    return float(pace_s) / float(factor)


def evolve_within_session(
    base_pace_s: float,
    lap_in_session: int,
    laps_in_session: int,
    total_evolution_pct: float = 0.5,
) -> float:
    """Apply intra-session grip-up: linear improvement from base_pace_s at
    lap 1 to base_pace_s × (1 - evolution) at the end. `total_evolution_pct`
    is the percentage improvement over the session.
    """
    if laps_in_session <= 1:
        return float(base_pace_s)
    progress = max(0.0, min(1.0, (lap_in_session - 1) / (laps_in_session - 1)))
    factor = 1.0 - (total_evolution_pct / 100.0) * progress
    return float(base_pace_s) * float(factor)
