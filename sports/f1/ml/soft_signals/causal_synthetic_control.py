"""Synthetic control — counterfactual analysis for "what if X drove Y's car?"

Construct a synthetic version of driver X using a weighted combination of other
drivers in the same car/era, fit on pre-treatment race data, then evaluate the
synthetic on post-treatment races.

Useful for: estimating mid-season swap effects (Pérez → Lawson at Red Bull),
rookie expected pace, "in a Mercedes Hamilton would be N tenths quicker" claims.

Implementation: Abadie et al. synthetic control method, with regularization
because driver-pool size is small (~20).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticControlFit:
    target_driver: str
    weights: dict[str, float]  # weight per donor driver
    pre_period_rmse: float


def fit_synthetic_control(
    panel: pd.DataFrame,
    target_driver: str,
    target_team: str,
    pre_treatment_races: list[str],
) -> SyntheticControlFit:
    """`panel` is long-format: (race, driver, pace_metric)."""
    raise NotImplementedError("convex weights minimizing pre-period MSE; sklearn QP solver")


def counterfactual_pace(fit: SyntheticControlFit, panel: pd.DataFrame, race: str) -> float:
    raise NotImplementedError
