"""Cox proportional hazards model for DNF — semiparametric survival.

No baseline-distribution assumption; estimates hazard ratios per covariate.
Robust when the lap-of-failure distribution doesn't fit a clean Weibull
(common when failures cluster early and late but not middle).

Trade-off: gives hazard ratios but no absolute hazard rate without an estimated
baseline. Use Breslow estimator for the baseline, then sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CoxFit:
    coef: dict[str, float]
    baseline_cumulative_hazard: pd.Series  # indexed by lap


def fit(stint_observations: pd.DataFrame) -> CoxFit:
    raise NotImplementedError("delegate to lifelines.CoxPHFitter")


def hazard_per_lap(fit_: CoxFit, lap: int, covariates: dict[str, float]) -> float:
    raise NotImplementedError
