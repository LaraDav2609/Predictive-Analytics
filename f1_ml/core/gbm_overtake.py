"""LightGBM overtake-completion head — alternative to events/overtake.py if you
want to ensemble.

Identical interface to events/overtake.py; this lives in core/ so it can be
trained with the rest of the production heads in one pipeline.

Features: gap_ahead_s, pace_delta_s, drs_available, tire_age_delta,
track_overtake_difficulty_index, lap_in_stint, sector_with_max_overtake_rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from f1_ml.common.registry import register


@register("core.gbm_overtake")
def build(**kwargs) -> "GBMOvertakeModel":
    return GBMOvertakeModel(**kwargs)


class GBMOvertakeModel:
    """Per-attempt probability of completing a pass on the car ahead."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.booster: LGBMClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMOvertakeModel":
        self.booster = LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
        )
        self.booster.fit(X, np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(pass succeeds) ∈ [0, 1] for each row."""
        if self.booster is None:
            raise RuntimeError("GBMOvertakeModel is not fitted; call .fit(X, y) first")
        proba = self.booster.predict_proba(X)
        return np.asarray(proba[:, 1])
