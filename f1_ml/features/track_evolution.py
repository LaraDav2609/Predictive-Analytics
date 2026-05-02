"""Track evolution / grip-up model — track surface gets faster Friday → Sunday
as rubber is laid down, and within each session as more laps accumulate.

Without this, FP3 pace is overestimated (it's faster than it would be in the race
on the same conditions because of grip evolution).

Fit a multiplicative grip factor per session that brings observed pace to a
common reference. Apply inverse when projecting practice pace onto race pace.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class EvolutionFit:
    track_code: str
    fp1_factor: float
    fp2_factor: float
    fp3_factor: float
    quali_factor: float
    race_factor: float = 1.0  # reference


def fit(historical_session_pace: pd.DataFrame) -> EvolutionFit:
    """Estimate session-to-session grip multipliers from historical races at this track."""
    raise NotImplementedError("per-track median pace ratio across sessions, smoothed")


def normalize_to_race(pace_s: float, session: str, fit_: EvolutionFit) -> float:
    """Convert a practice / quali pace estimate to its race-equivalent."""
    raise NotImplementedError
