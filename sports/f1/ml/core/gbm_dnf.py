"""LightGBM DNF head — per-lap binary classifier for "did this car retire on this lap?"

Alternative to the parametric Weibull / Cox AFT models in events/. GBM is more
flexible (catches feature interactions) but loses the interpretable hazard ratios.

Use as an ensemble partner: GBM + Weibull averaged tends to calibrate better
than either alone. Class imbalance is severe (~2-3 % of laps end in DNF), so
we set `class_weight='balanced'` to keep the loss informative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light CI
    LGBMClassifier = None

from common.ml.registry import register


@register("core.gbm_dnf")
def build(**kwargs) -> "GBMDNFModel":
    return GBMDNFModel(**kwargs)


class GBMDNFModel:
    """Per-lap probability of mechanical / accident retirement."""

    def __init__(
        self,
        n_estimators: int = 800,
        learning_rate: float = 0.02,
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

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMDNFModel":
        """Binary objective with balanced class weights to compensate for the
        ~2-3 % positive rate."""
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is required to fit GBMDNFModel; install lightgbm or use a deterministic DNF fallback.")
        self.booster = LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            class_weight="balanced",
            random_state=self.random_state,
            verbose=-1,
        )
        self.booster.fit(X, np.asarray(y).astype(int))
        return self

    def hazard_per_lap(self, features: pd.DataFrame) -> np.ndarray:
        """Returns P(DNF on this lap) ∈ [0, 1] — the simulator multiplies this
        with `dt` (=1 lap) to draw a Bernoulli at each tick."""
        if self.booster is None:
            raise RuntimeError("GBMDNFModel is not fitted; call .fit(X, y) first")
        proba = self.booster.predict_proba(features)
        # Multi-column predict_proba even for binary; column index 1 = positive class.
        return np.asarray(proba[:, 1])
