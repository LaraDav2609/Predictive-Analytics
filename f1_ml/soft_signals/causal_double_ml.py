"""Double / debiased machine learning for car-vs-driver effect estimation.

The hardest causal question in F1: of a driver's pace, how much is car and
how much is driver? Naive regression of pace on driver dummies confounds the
two because top drivers cluster in top cars.

Double ML (Chernozhukov 2018) gives a consistent driver-effect estimate by:
1. Predicting pace from car-only features (machine A).
2. Predicting driver-indicator from car-only features (machine B).
3. Regressing residuals_A on residuals_B → debiased driver effect.

Use sample-splitting to avoid overfit-bias.

Output: per-driver causal-effect estimate that we trust more than raw pace
deltas. Feeds into hierarchical_bayes priors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def double_ml_driver_effect(
    pace_data: pd.DataFrame,
    driver_dummies: pd.DataFrame,
    car_features: pd.DataFrame,
    n_folds: int = 5,
) -> pd.Series:
    """Returns per-driver causal effect (s/lap relative to baseline driver)."""
    raise NotImplementedError(
        "cross-fit residualization with LightGBM nuisance learners; OLS on residuals"
    )
