"""LightGBM overtake-completion head — alternative to events/overtake.py if you
want to ensemble.

Identical interface to events/overtake.py; this lives in core/ so it can be
trained with the rest of the production heads in one pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1_ml.common.registry import register


@register("core.gbm_overtake")
def build(**kwargs) -> "GBMOvertakeModel":
    return GBMOvertakeModel(**kwargs)


class GBMOvertakeModel:
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.03) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.booster = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMOvertakeModel":
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
