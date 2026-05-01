"""LightGBM pace head — predicts mean lap pace given engineered features.

Features: hierarchical_bayes posterior mean + tire_age + compound + fuel_kg +
dirty_air_seconds + track_evolution_factor + weather + driver_form.

Used inside the simulator to set the per-lap pace mean for each driver. The
hierarchical model gives a principled prior; the GBM captures non-linearities
the linear hierarchical pace term misses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_ml.common.registry import register


@register("core.gbm_pace")
def build(**kwargs) -> "GBMPaceModel":
    return GBMPaceModel(**kwargs)


class GBMPaceModel:
    def __init__(self, n_estimators: int = 1000, learning_rate: float = 0.02, num_leaves: int = 63) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.booster = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMPaceModel":
        raise NotImplementedError("LGBMRegressor; objective='regression_l1' for robustness to traffic-spike outliers")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_with_quantiles(self, X: pd.DataFrame, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)) -> np.ndarray:
        """Quantile regression mode — train one model per alpha, return (n, len(alphas))."""
        raise NotImplementedError
