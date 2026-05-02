"""LightGBM DNF head — per-lap binary classifier for "did this car retire on this lap?"

Alternative to the parametric Weibull / Cox AFT models in events/. GBM is more
flexible (catches feature interactions) but loses the interpretable hazard ratios.

Use as an ensemble partner: GBM + Weibull averaged tends to calibrate better
than either alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_ml.common.registry import register


@register("core.gbm_dnf")
def build(**kwargs) -> "GBMDNFModel":
    return GBMDNFModel(**kwargs)


class GBMDNFModel:
    def __init__(self, n_estimators: int = 800, learning_rate: float = 0.02) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.booster = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMDNFModel":
        raise NotImplementedError("LGBMClassifier with class_weight='balanced'; DNFs are ~2-3% of laps")

    def hazard_per_lap(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
