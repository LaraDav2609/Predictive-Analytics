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
from sklearn.linear_model import LogisticRegression


class StackedEnsemble:
    """Meta-learner over base-model OOF predictions.

    `meta_learner` selects between:
      - 'logistic': sklearn LogisticRegression (well-calibrated when bases are good).
      - 'lightgbm': monotonic-constrained LGBMClassifier (catches non-linear blends).
    """

    def __init__(
        self,
        base_model_names: list[str],
        meta_learner: str = "logistic",
        random_state: int = 42,
    ) -> None:
        if meta_learner not in ("logistic", "lightgbm"):
            raise ValueError(f"meta_learner must be 'logistic' or 'lightgbm', got '{meta_learner}'")
        self.base_model_names = list(base_model_names)
        self.meta_learner_kind = meta_learner
        self.random_state = random_state
        self.meta_model = None
        self.feature_columns: list[str] = []

    def fit(self, oof_predictions: pd.DataFrame, outcomes: np.ndarray) -> "StackedEnsemble":
        """`oof_predictions` columns: one per base model + (optional) context features.

        Each base model column is a probability ∈ [0, 1]. Outcomes are binary 0/1.
        """
        missing = [c for c in self.base_model_names if c not in oof_predictions.columns]
        if missing:
            raise KeyError(f"oof_predictions missing base columns: {missing}")
        self.feature_columns = list(oof_predictions.columns)
        X = oof_predictions[self.feature_columns].to_numpy(dtype=float)
        y = np.asarray(outcomes, dtype=int)

        if self.meta_learner_kind == "logistic":
            self.meta_model = LogisticRegression(
                C=1.0, max_iter=1000, random_state=self.random_state
            )
            self.meta_model.fit(X, y)
        else:
            from lightgbm import LGBMClassifier
            # Monotonic constraint per base column = +1 (higher base prob ⇒
            # higher meta prob). Context features unconstrained (=0).
            monotone = [1 if c in self.base_model_names else 0
                        for c in self.feature_columns]
            self.meta_model = LGBMClassifier(
                objective="binary",
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                monotone_constraints=monotone,
                random_state=self.random_state,
                verbose=-1,
            )
            self.meta_model.fit(X, y)
        return self

    def predict_proba(self, base_predictions: pd.DataFrame) -> np.ndarray:
        if self.meta_model is None:
            raise RuntimeError("StackedEnsemble is not fitted; call .fit(...) first")
        X = base_predictions.reindex(columns=self.feature_columns, fill_value=0.0).to_numpy(dtype=float)
        return np.asarray(self.meta_model.predict_proba(X)[:, 1])
