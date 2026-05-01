"""Tire degradation curve fitting — lap-time-vs-tire-age regression per stint.

Models: linear-with-cliff (most stints) and exponential decay (high-deg compounds).
Robust regression (Huber / RANSAC) to handle outlier laps from traffic, lockups,
or yellow-flag sectors.

Outputs per (driver, compound, track) a degradation rate (s/lap) the simulator
applies during MC rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DegradationFit:
    driver_code: str
    compound: str
    track_code: str
    base_pace_s: float       # extrapolated lap time at tire_age=0
    deg_per_lap_s: float     # linear coefficient
    cliff_lap: int | None    # None if no cliff observed; else lap where deg accelerates
    cliff_factor: float      # multiplier on deg after cliff
    n_observations: int


def fit_linear(
    stint_laps: pd.DataFrame,
    tire_age_col: str = "tire_age_laps",
    pace_col: str = "fuel_corrected_lap_time_s",
    driver_col: str = "driver_code",
    compound_col: str = "compound",
    track_col: str = "track_code",
) -> DegradationFit:
    """Robust linear regression of pace on tire age. Fall-back when there's
    not enough data for a cliff search.

    Use Huber regression to dampen outlier laps (traffic, lockups, yellow
    sectors) that would otherwise inflate the slope estimate.
    """
    from sklearn.linear_model import HuberRegressor

    if len(stint_laps) < 4:
        # Too little data to fit; return a sensible default with the mean as base pace.
        mean_pace = float(stint_laps[pace_col].mean()) if len(stint_laps) else 0.0
        return DegradationFit(
            driver_code=str(stint_laps[driver_col].iloc[0]) if len(stint_laps) else "",
            compound=str(stint_laps[compound_col].iloc[0]) if len(stint_laps) else "",
            track_code=str(stint_laps[track_col].iloc[0]) if track_col in stint_laps.columns and len(stint_laps) else "",
            base_pace_s=mean_pace,
            deg_per_lap_s=0.0,
            cliff_lap=None,
            cliff_factor=1.0,
            n_observations=len(stint_laps),
        )

    X = stint_laps[[tire_age_col]].to_numpy(dtype=float)
    y = stint_laps[pace_col].to_numpy(dtype=float)
    model = HuberRegressor().fit(X, y)
    return DegradationFit(
        driver_code=str(stint_laps[driver_col].iloc[0]),
        compound=str(stint_laps[compound_col].iloc[0]),
        track_code=str(stint_laps[track_col].iloc[0]) if track_col in stint_laps.columns else "",
        base_pace_s=float(model.intercept_),
        deg_per_lap_s=float(model.coef_[0]),
        cliff_lap=None,
        cliff_factor=1.0,
        n_observations=len(stint_laps),
    )


def fit_linear_with_cliff(
    stint_laps: pd.DataFrame,
    tire_age_col: str = "tire_age_laps",
    pace_col: str = "fuel_corrected_lap_time_s",
    min_segment_laps: int = 4,
    cliff_significance_s: float = 0.15,
    **kwargs,
) -> DegradationFit:
    """Robust two-segment regression with a learned breakpoint.

    Search every plausible breakpoint and pick the one whose post-cliff slope
    is at least `cliff_significance_s` higher than pre-cliff. If no significant
    cliff exists, falls back to plain linear.
    """
    base_fit = fit_linear(stint_laps, tire_age_col=tire_age_col, pace_col=pace_col, **kwargs)
    n = len(stint_laps)
    if n < 2 * min_segment_laps:
        return base_fit

    sorted_laps = stint_laps.sort_values(tire_age_col).reset_index(drop=True)
    best_cliff_lap: int | None = None
    best_cliff_factor: float = 1.0
    best_extra_slope = cliff_significance_s

    for cut in range(min_segment_laps, n - min_segment_laps):
        pre = sorted_laps.iloc[:cut]
        post = sorted_laps.iloc[cut:]
        pre_fit = fit_linear(pre, tire_age_col=tire_age_col, pace_col=pace_col, **kwargs)
        post_fit = fit_linear(post, tire_age_col=tire_age_col, pace_col=pace_col, **kwargs)
        extra_slope = post_fit.deg_per_lap_s - pre_fit.deg_per_lap_s
        if extra_slope > best_extra_slope and pre_fit.deg_per_lap_s != 0:
            best_extra_slope = extra_slope
            best_cliff_lap = int(sorted_laps[tire_age_col].iloc[cut])
            best_cliff_factor = post_fit.deg_per_lap_s / max(pre_fit.deg_per_lap_s, 1e-6)

    return DegradationFit(
        driver_code=base_fit.driver_code,
        compound=base_fit.compound,
        track_code=base_fit.track_code,
        base_pace_s=base_fit.base_pace_s,
        deg_per_lap_s=base_fit.deg_per_lap_s,
        cliff_lap=best_cliff_lap,
        cliff_factor=best_cliff_factor,
        n_observations=base_fit.n_observations,
    )


def fit_exponential(stint_laps: pd.DataFrame, tire_age_col: str = "tire_age_laps",
                    pace_col: str = "fuel_corrected_lap_time_s") -> DegradationFit:
    """For high-deg / soft compounds where decay is non-linear.
    Fits pace = base + a * exp(b * tire_age) by linearizing around the residual.
    """
    import numpy as np
    from scipy.optimize import least_squares

    if len(stint_laps) < 5:
        return fit_linear(stint_laps, tire_age_col=tire_age_col, pace_col=pace_col)

    age = stint_laps[tire_age_col].to_numpy(dtype=float)
    pace = stint_laps[pace_col].to_numpy(dtype=float)

    def residual(params):
        base, a, b = params
        return pace - (base + a * np.exp(b * age))

    p0 = [pace.min(), 0.1, 0.05]
    try:
        result = least_squares(residual, p0, method="trf")
        base, a, b = result.x
    except Exception:
        return fit_linear(stint_laps, tire_age_col=tire_age_col, pace_col=pace_col)

    return DegradationFit(
        driver_code=str(stint_laps["driver_code"].iloc[0]) if "driver_code" in stint_laps.columns else "",
        compound=str(stint_laps["compound"].iloc[0]) if "compound" in stint_laps.columns else "",
        track_code=str(stint_laps["track_code"].iloc[0]) if "track_code" in stint_laps.columns else "",
        base_pace_s=float(base),
        deg_per_lap_s=float(a * b),  # initial slope = a*b
        cliff_lap=None,
        cliff_factor=float(b),  # use cliff_factor field to carry exponential rate
        n_observations=len(stint_laps),
    )


def predict_pace_at_age(fit: DegradationFit, tire_age: int) -> float:
    """Lap-time prediction for a given tire age, applying cliff if past breakpoint."""
    base = fit.base_pace_s
    if fit.cliff_lap is None or tire_age <= fit.cliff_lap:
        return base + fit.deg_per_lap_s * tire_age
    pre_cliff_pace = base + fit.deg_per_lap_s * fit.cliff_lap
    post_cliff_age = tire_age - fit.cliff_lap
    return pre_cliff_pace + fit.deg_per_lap_s * fit.cliff_factor * post_cliff_age
