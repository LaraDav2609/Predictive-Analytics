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

import numpy as np
import pandas as pd


@dataclass
class SafetyCarRate:
    track_code: str
    base_rate_per_lap: float       # base Poisson rate per lap (race-pace conditions)
    rain_multiplier: float          # multiplier when rain_intensity > 0.3
    late_race_multiplier: float     # last 25% of laps
    typical_duration_laps: int


def fit(
    historical_sc_events: pd.DataFrame,
    track_col: str = "track_code",
    laps_col: str = "race_laps",
    events_col: str = "sc_events",
    rain_events_col: str = "sc_events_rain",
    rain_laps_col: str = "rain_laps",
    late_events_col: str = "sc_events_late",
    late_laps_col: str = "late_laps",
    duration_col: str = "avg_sc_duration_laps",
) -> dict[str, SafetyCarRate]:
    """Per-track rates from historical SC / VSC counts.

    Expected schema (one row per race): track_code, race_laps, sc_events, and
    optionally sc_events_rain / rain_laps / sc_events_late / late_laps /
    avg_sc_duration_laps. Missing optional cols default to neutral
    multipliers (=1).
    """
    if historical_sc_events.empty:
        return {}

    out: dict[str, SafetyCarRate] = {}
    for track, grp in historical_sc_events.groupby(track_col):
        events = float(grp[events_col].sum())
        laps = float(grp[laps_col].sum())
        # Add Bayesian smoothing (1 event per 100 laps prior) to avoid zero-rate
        # tracks producing degenerate samples.
        base_rate = (events + 1.0) / (laps + 100.0)

        rain_mult = _safe_multiplier(
            grp, rain_events_col, rain_laps_col, base_rate
        )
        late_mult = _safe_multiplier(
            grp, late_events_col, late_laps_col, base_rate
        )
        duration = (
            float(grp[duration_col].mean())
            if duration_col in grp.columns and grp[duration_col].notna().any()
            else 4.0  # F1 SC neutralization median
        )
        out[str(track)] = SafetyCarRate(
            track_code=str(track),
            base_rate_per_lap=base_rate,
            rain_multiplier=rain_mult,
            late_race_multiplier=late_mult,
            typical_duration_laps=int(round(duration)),
        )
    return out


def _safe_multiplier(
    grp: pd.DataFrame,
    events_col: str,
    laps_col: str,
    base_rate: float,
) -> float:
    """Return conditional rate / base rate, default 1.0 if data missing."""
    if (
        events_col not in grp.columns
        or laps_col not in grp.columns
        or grp[laps_col].sum() == 0
    ):
        return 1.0
    cond_events = float(grp[events_col].sum())
    cond_laps = float(grp[laps_col].sum())
    if cond_laps == 0 or base_rate == 0:
        return 1.0
    cond_rate = (cond_events + 0.5) / (cond_laps + 50.0)
    return float(max(0.1, cond_rate / base_rate))


def sample_event(
    rate: SafetyCarRate,
    lap: int,
    total_laps: int,
    rain_intensity: float,
    rng: np.random.Generator | None = None,
) -> bool:
    """Bernoulli draw at the (state-modified) per-lap rate."""
    if total_laps <= 0:
        return False
    rng = rng or np.random.default_rng()
    rate_eff = rate.base_rate_per_lap
    if rain_intensity > 0.3:
        rate_eff *= rate.rain_multiplier
    if lap >= int(total_laps * 0.75):
        rate_eff *= rate.late_race_multiplier
    # Convert the rate (events / lap) to per-lap probability via Poisson:
    # P(at least one event in this lap) = 1 - exp(-λ)
    p = 1.0 - float(np.exp(-rate_eff))
    return bool(rng.uniform() < p)


def expected_events(rate: SafetyCarRate, total_laps: int) -> float:
    """Expected number of SC events over a full race — diagnostic."""
    return float(rate.base_rate_per_lap * total_laps)
