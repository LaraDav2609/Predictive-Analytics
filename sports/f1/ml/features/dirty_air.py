"""Dirty-air / following penalty — quantify how much lap time inflates when a car
is in the wake of another.

Empirical: when gap-ahead < ~1.5 s, lap time inflates 0.3-1.5 s depending on track
aerodynamics (Monaco/Hungary high; Monza/Spa lower). Critical input to the simulator's
overtake module — without it, races where chasers are stuck behind look impossible.

Estimate from clean-air vs. dirty-air pairs in historical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DirtyAirModel:
    track_code: str
    threshold_s: float       # gap-ahead at which dirty air kicks in
    penalty_per_s_close: float  # added lap-time per second of gap closure
    max_penalty_s: float


def fit(
    laps_with_gaps: pd.DataFrame,
    gap_threshold_s: float = 1.5,
    min_clean_air_laps: int = 5,
    track_col: str = "track_code",
) -> DirtyAirModel:
    """Compare lap times in clean air (gap_ahead > threshold) vs. close (< threshold)
    for the same driver+compound+tire_age bucket.

    Strategy:
      1. Per (driver, compound, tire_age_bucket), compute clean-air baseline pace
         from laps with gap_ahead > threshold.
      2. For laps with gap_ahead <= threshold, compute delta_lap_time = lap_time - baseline.
      3. Regress delta on (threshold - gap_ahead) → penalty per second of gap closure.
    """
    df = laps_with_gaps.dropna(subset=["gap_ahead_s", "lap_time_s"]).copy()
    if df.empty:
        return DirtyAirModel(track_code="", threshold_s=gap_threshold_s,
                             penalty_per_s_close=0.0, max_penalty_s=0.0)

    df["tire_age_bucket"] = (df["tire_age_laps"] // 3).astype(int)
    track_code = str(df[track_col].iloc[0]) if track_col in df.columns else ""

    clean = df[df["gap_ahead_s"] > gap_threshold_s].groupby(
        ["driver_code", "compound", "tire_age_bucket"]
    )["lap_time_s"].mean().rename("clean_baseline_s")

    if len(clean) < min_clean_air_laps:
        return DirtyAirModel(track_code=track_code, threshold_s=gap_threshold_s,
                             penalty_per_s_close=0.0, max_penalty_s=0.0)

    df = df.join(clean, on=["driver_code", "compound", "tire_age_bucket"])
    dirty = df[(df["gap_ahead_s"] <= gap_threshold_s) & df["clean_baseline_s"].notna()].copy()
    if len(dirty) < 5:
        return DirtyAirModel(track_code=track_code, threshold_s=gap_threshold_s,
                             penalty_per_s_close=0.0, max_penalty_s=0.0)

    dirty["closeness"] = (gap_threshold_s - dirty["gap_ahead_s"]).clip(lower=0.0)
    dirty["delta_s"] = dirty["lap_time_s"] - dirty["clean_baseline_s"]

    # Robust slope via Huber on (closeness, delta_s).
    from sklearn.linear_model import HuberRegressor
    X = dirty[["closeness"]].to_numpy(dtype=float)
    y = dirty["delta_s"].to_numpy(dtype=float)
    try:
        model = HuberRegressor().fit(X, y)
        slope = float(model.coef_[0])
    except Exception:
        slope = float((dirty["delta_s"] / dirty["closeness"].clip(lower=0.05)).median())

    slope = max(0.0, slope)  # negative slope is unphysical → clamp
    return DirtyAirModel(
        track_code=track_code,
        threshold_s=gap_threshold_s,
        penalty_per_s_close=slope,
        max_penalty_s=slope * gap_threshold_s,
    )


def apply_penalty(model: DirtyAirModel, gap_ahead_s: float) -> float:
    """Lap-time penalty (seconds) added when running this close to the car ahead."""
    if gap_ahead_s >= model.threshold_s:
        return 0.0
    closeness = max(0.0, model.threshold_s - gap_ahead_s)
    return min(model.max_penalty_s, model.penalty_per_s_close * closeness)
