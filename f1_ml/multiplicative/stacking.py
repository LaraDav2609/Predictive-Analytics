"""Stacked ensemble — meta-learner blends predictions from physics-MC, GBM heads,
and sequence models.

Each base model produces a probability per market. A meta-learner (logistic
regression or shallow GBM) takes those probabilities + a few context features
(track type, weather, race-progress %) and outputs a final blended probability.

Why stacking over simple averaging: different base models are best in different
regimes (physics-MC handles weather chaos; sequence models nail driver-form
trends). The meta-learner learns when to trust which.

Train on out-of-fold predictions to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StackedEnsemble:
    def __init__(self, base_model_names: list[str], meta_learner: str = "logistic") -> None:
        self.base_model_names = base_model_names
        self.meta_learner_kind = meta_learner
        self.meta_model = None

    def fit(self, oof_predictions: pd.DataFrame, outcomes: np.ndarray) -> "StackedEnsemble":
        """`oof_predictions` columns: one per base model + context features."""
        raise NotImplementedError("LogisticRegression or LightGBM with monotonic constraints")

    def predict_proba(self, base_predictions: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
