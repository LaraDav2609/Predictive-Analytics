"""Weibull Accelerated Failure Time (AFT) model for DNF — parametric survival.

Models time-to-failure (in laps) as Weibull with shape k and scale lambda that
depend on covariates (peak G's, brake-temp proxy from deceleration intensity,
engine mileage, lap, ambient temp).

Why Weibull AFT: closed-form hazard, easy interpretation (acceleration factor
per covariate), works well with the small DNF count per season (~30-50).

Use Cox PH (dnf_cox.py) when you'd rather not commit to a parametric distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WeibullAFTFit:
    coef: dict[str, float]
    shape_k: float
    scale_lambda: float


def fit(stint_observations: pd.DataFrame) -> WeibullAFTFit:
    """`stint_observations` columns: laps_completed, dnf (0/1), + covariates."""
    raise NotImplementedError("delegate to lifelines.WeibullAFTFitter")


def hazard_per_lap(fit_: WeibullAFTFit, lap: int, covariates: dict[str, float]) -> float:
    """Instantaneous hazard at this lap given covariates — what the simulator samples."""
    raise NotImplementedError
