"""Weibull Accelerated Failure Time (AFT) model for DNF — parametric survival.

Models time-to-failure (in laps) as Weibull with shape k and scale lambda that
depend on covariates (peak G's, brake-temp proxy from deceleration intensity,
engine mileage, lap, ambient temp).

Why Weibull AFT: closed-form hazard, easy interpretation (acceleration factor
per covariate), works well with the small DNF count per season (~30-50).

Use Cox PH (dnf_cox.py) when you'd rather not commit to a parametric distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from lifelines import WeibullAFTFitter
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light CI
    WeibullAFTFitter = None


@dataclass
class WeibullAFTFit:
    """Fitted Weibull AFT model. Holds the lifelines fitter for prediction
    and a copy of summary coefficients for inspection / serialization."""
    coef: dict[str, float]
    shape_k: float
    scale_lambda: float
    fitter: Any = field(default=None, repr=False)
    feature_columns: list[str] = field(default_factory=list)


def fit(
    stint_observations: pd.DataFrame,
    duration_col: str = "laps_completed",
    event_col: str = "dnf",
) -> WeibullAFTFit:
    """Fit Weibull AFT on stint-level observations.

    `stint_observations` must contain `duration_col` (laps until DNF or
    censoring), `event_col` (1 if DNF observed, 0 if censored), and any
    number of numeric covariate columns.
    """
    if stint_observations.empty:
        raise ValueError("stint_observations is empty")
    if duration_col not in stint_observations.columns:
        raise KeyError(f"missing duration column '{duration_col}'")
    if event_col not in stint_observations.columns:
        raise KeyError(f"missing event column '{event_col}'")

    if WeibullAFTFitter is None:
        raise RuntimeError("lifelines is required to fit WeibullAFT; install lifelines or use the deterministic DNF fallback.")

    aft = WeibullAFTFitter(penalizer=0.01)
    # lifelines requires duration_col & event_col passed by name
    aft.fit(stint_observations, duration_col=duration_col, event_col=event_col)

    # Pull learned parameters. lifelines parameterizes Weibull as
    # S(t) = exp(-(t/λ)^ρ) with λ depending on covariates via AFT.
    params = aft.params_
    # The intercept of the lambda submodel: exp(intercept) = baseline scale.
    lambda_intercept = float(params.xs("Intercept", level=1)["lambda_"])
    rho_intercept = float(params.xs("Intercept", level=1)["rho_"])
    scale_lambda = float(np.exp(lambda_intercept))
    shape_k = float(np.exp(rho_intercept))

    coef_series = params.xs("lambda_", level=0)
    coef = {str(k): float(v) for k, v in coef_series.items() if k != "Intercept"}

    feature_cols = [c for c in stint_observations.columns
                    if c not in (duration_col, event_col)]
    return WeibullAFTFit(
        coef=coef,
        shape_k=shape_k,
        scale_lambda=scale_lambda,
        fitter=aft,
        feature_columns=feature_cols,
    )


def hazard_per_lap(
    fit_: WeibullAFTFit,
    lap: int,
    covariates: dict[str, float],
) -> float:
    """Instantaneous hazard at this lap given covariates — what the simulator
    samples to flip a Bernoulli per lap.

    Returns a probability in [0, 1) — the integrated hazard over a single lap
    interval, capped to keep Bernoulli draws well-defined.
    """
    if fit_.fitter is None:
        raise RuntimeError("WeibullAFT is not fitted; call fit() first")
    lap = int(lap)
    if lap < 1:
        return 0.0
    row = pd.DataFrame([{c: covariates.get(c, 0.0) for c in fit_.feature_columns}])
    # Cumulative hazard at lap t and t-1; their difference ≈ integrated
    # hazard over the lap interval, ≈ P(failure | survived to t-1) for small
    # increments. Cap at 1.0 - ε for numerical safety.
    times = np.array([max(0.5, lap - 1.0), float(lap)])
    cumhz = fit_.fitter.predict_cumulative_hazard(row, times=times).values.ravel()
    delta = float(cumhz[1] - cumhz[0])
    if delta < 0.0:
        delta = 0.0
    # Convert integrated hazard to per-interval failure probability:
    # P(fail in interval | survived) = 1 - exp(-Δ)
    return float(1.0 - np.exp(-delta))
