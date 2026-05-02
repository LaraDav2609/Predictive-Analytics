"""Safety car / VSC arrival as a track-specific Poisson process.

Track-conditional rates differ wildly: Monaco / Baku / Singapore see ~70%+ of
races run with at least one SC; Silverstone / Spa under 25%. State-dependent
extension: rate inflates with rain intensity, increases late in race as cars/
brakes get tired.

The simulator samples SC events per lap from this; when triggered, it bunches
the field and re-runs an overtake roll on the restart lap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SafetyCarRate:
    track_code: str
    base_rate_per_lap: float       # base Poisson rate
    rain_multiplier: float          # multiplier when rain_intensity > 0.3
    late_race_multiplier: float     # last 25% of laps
    typical_duration_laps: int


def fit(historical_sc_events: pd.DataFrame) -> dict[str, SafetyCarRate]:
    """Per-track rates from historical SC / VSC counts."""
    raise NotImplementedError


def sample_event(rate: SafetyCarRate, lap: int, total_laps: int, rain_intensity: float) -> bool:
    """Bernoulli draw at the (state-modified) per-lap rate."""
    raise NotImplementedError
