"""Cox proportional hazards model for DNF — semiparametric survival.

No baseline-distribution assumption; estimates hazard ratios per covariate.
Robust when the lap-of-failure distribution doesn't fit a clean Weibull
(common when failures cluster early and late but not middle).

Trade-off: gives hazard ratios but no absolute hazard rate without an estimated
baseline. Uses the Breslow estimator for the baseline cumulative hazard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light CI
    CoxPHFitter = None


@dataclass
class CoxFit:
    coef: dict[str, float]
    baseline_cumulative_hazard: pd.Series  # indexed by lap (Breslow)
    fitter: Any = field(default=None, repr=False)
    feature_columns: list[str] = field(default_factory=list)


def fit(
    stint_observations: pd.DataFrame,
    duration_col: str = "laps_completed",
    event_col: str = "dnf",
) -> CoxFit:
    if stint_observations.empty:
        raise ValueError("stint_observations is empty")

    if CoxPHFitter is None:
        raise RuntimeError("lifelines is required to fit CoxPH; install lifelines or use the deterministic DNF fallback.")

    cox = CoxPHFitter(penalizer=0.01)
    cox.fit(stint_observations, duration_col=duration_col, event_col=event_col)

    coef = {str(k): float(v) for k, v in cox.params_.items()}
    bch = cox.baseline_cumulative_hazard_
    # CoxPHFitter returns a single-column DataFrame; squeeze to Series.
    bch_series = bch.iloc[:, 0] if isinstance(bch, pd.DataFrame) else bch
    bch_series.index.name = "lap"

    feature_cols = [c for c in stint_observations.columns
                    if c not in (duration_col, event_col)]
    return CoxFit(
        coef=coef,
        baseline_cumulative_hazard=bch_series,
        fitter=cox,
        feature_columns=feature_cols,
    )


def hazard_per_lap(
    fit_: CoxFit,
    lap: int,
    covariates: dict[str, float],
) -> float:
    """Per-lap failure probability under the Cox model.

    Uses h(t|X) = h_0(t) * exp(β'X). Baseline hazard at lap t is approximated
    as ΔH_0(t) = H_0(t) - H_0(t-1) from the Breslow estimator.
    """
    if fit_.fitter is None:
        raise RuntimeError("CoxPH is not fitted; call fit() first")
    lap = int(lap)
    if lap < 1:
        return 0.0

    bch = fit_.baseline_cumulative_hazard
    # Step interpolation to lap t.
    h_curr = float(_step_lookup(bch, lap))
    h_prev = float(_step_lookup(bch, lap - 1)) if lap > 1 else 0.0
    delta_h0 = max(h_curr - h_prev, 0.0)

    log_partial = sum(
        fit_.coef.get(c, 0.0) * float(covariates.get(c, 0.0))
        for c in fit_.feature_columns
    )
    integrated = delta_h0 * float(np.exp(log_partial))
    return float(1.0 - np.exp(-integrated))


def _step_lookup(series: pd.Series, t: float) -> float:
    """Right-continuous step function lookup: largest index ≤ t."""
    if len(series) == 0:
        return 0.0
    idx = series.index.values
    # Indices ≤ t
    mask = idx <= t
    if not mask.any():
        return 0.0
    return float(series.values[mask][-1])
