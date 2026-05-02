"""LightGBM pace head — predicts mean lap pace given engineered features.

Features: hierarchical_bayes posterior mean + tire_age + compound + fuel_kg +
dirty_air_seconds + track_evolution_factor + weather + driver_form.

Used inside the simulator to set the per-lap pace mean for each driver. The
hierarchical model gives a principled prior; the GBM captures non-linearities
the linear hierarchical pace term misses.

Two prediction modes:
  - `fit` / `predict` — point estimate using L1-regression objective (robust
    to traffic spikes / lockup outliers).
  - `fit_quantiles` / `predict_with_quantiles` — one booster per requested
    alpha, returns a stacked (n_rows, n_alphas) matrix. Used inside the
    simulator to sample lap times rather than just take the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from f1_ml.common.registry import register


@register("core.gbm_pace")
def build(**kwargs) -> "GBMPaceModel":
    return GBMPaceModel(**kwargs)


class GBMPaceModel:
    """Lap-pace regressor. L1 objective by default (robust to outliers)."""

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.02,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.booster: LGBMRegressor | None = None
        self.quantile_boosters: dict[float, LGBMRegressor] = {}

    # ------------------------------------------------------------------ point
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBMPaceModel":
        """Fit a single L1-regression GBM on (X, y)."""
        self.booster = LGBMRegressor(
            objective="regression_l1",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            verbose=-1,
        )
        self.booster.fit(X, np.asarray(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("GBMPaceModel is not fitted; call .fit(X, y) first")
        return np.asarray(self.booster.predict(X))

    # -------------------------------------------------------------- quantile
    def fit_quantiles(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> "GBMPaceModel":
        """Train one booster per α with `objective='quantile'`. Boosters are
        independent and may not be perfectly monotone in α; the simulator can
        tolerate occasional crossings."""
        y = np.asarray(y)
        self.quantile_boosters = {}
        for a in alphas:
            mdl = LGBMRegressor(
                objective="quantile",
                alpha=float(a),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                random_state=self.random_state,
                verbose=-1,
            )
            mdl.fit(X, y)
            self.quantile_boosters[float(a)] = mdl
        return self

    def predict_with_quantiles(
        self,
        X: pd.DataFrame,
        alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> np.ndarray:
        """Returns shape (n, len(alphas)). Requires `fit_quantiles` first."""
        if not self.quantile_boosters:
            raise RuntimeError(
                "GBMPaceModel has no quantile boosters; call .fit_quantiles(X, y, alphas) first"
            )
        cols = []
        for a in alphas:
            key = float(a)
            if key not in self.quantile_boosters:
                raise KeyError(
                    f"alpha={a} not fitted; available: {sorted(self.quantile_boosters.keys())}"
                )
            cols.append(np.asarray(self.quantile_boosters[key].predict(X)))
        return np.stack(cols, axis=1)
