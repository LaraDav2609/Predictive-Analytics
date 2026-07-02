"""Per-lap overtake-attempt probability — gradient-boosted classifier.

Inputs: gap_ahead_s, pace_delta_s_per_lap, drs_available, tire_age_delta_laps,
compound_advantage, track_overtake_difficulty_index (Monaco ~ 0.05, Baku ~ 0.7).

Output: P(pass completed this lap | features). The simulator rolls this every
lap for cars within DRS range (~1.0 s).

LightGBM is the workhorse here — interactions are highly non-linear (DRS only
matters when the chaser is in DRS range AND has pace).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:  # pragma: no cover - exercised in dependency-light CI
    LGBMClassifier = None


@dataclass
class OvertakeFeatures:
    gap_ahead_s: float
    pace_delta_s: float
    drs_available: bool
    tire_age_delta_laps: int
    compound_advantage: int  # softer compound = +1, harder = -1, same = 0
    track_overtake_index: float

    def as_row(self) -> dict[str, float]:
        d = asdict(self)
        d["drs_available"] = float(bool(self.drs_available))
        return d


class OvertakeModel:
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
        self.feature_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OvertakeModel":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is required to fit OvertakeModel; install lightgbm or use a deterministic overtake fallback.")
        self.feature_columns = list(X.columns)
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

    def predict_proba(self, features: OvertakeFeatures | pd.DataFrame) -> float | np.ndarray:
        """Pass a single OvertakeFeatures → returns scalar P(pass).
        Pass a DataFrame → returns ndarray of P(pass) per row.
        """
        if self.booster is None:
            raise RuntimeError("OvertakeModel is not fitted; call .fit(X, y) first")
        if isinstance(features, OvertakeFeatures):
            row_df = pd.DataFrame([features.as_row()])
            if self.feature_columns:
                row_df = row_df.reindex(columns=self.feature_columns, fill_value=0.0)
            return float(self.booster.predict_proba(row_df)[0, 1])
        return np.asarray(self.booster.predict_proba(features)[:, 1])
