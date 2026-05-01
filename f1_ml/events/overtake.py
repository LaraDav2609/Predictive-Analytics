"""Per-lap overtake-attempt probability — gradient-boosted classifier.

Inputs: gap_ahead_s, pace_delta_s_per_lap, drs_available, tire_age_delta_laps,
compound_advantage, track_overtake_difficulty_index (Monaco ~ 0.05, Baku ~ 0.7).

Output: P(pass completed this lap | features). The simulator rolls this every
lap for cars within DRS range (~1.0 s).

LightGBM is the workhorse here — interactions are highly non-linear (DRS only
matters when the chaser is in DRS range AND has pace).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OvertakeFeatures:
    gap_ahead_s: float
    pace_delta_s: float
    drs_available: bool
    tire_age_delta_laps: int
    compound_advantage: int  # softer compound = +1, harder = -1, same = 0
    track_overtake_index: float


class OvertakeModel:
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.03) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.booster = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "OvertakeModel":
        raise NotImplementedError("LGBMClassifier; class weight 'balanced'; AUC ~0.85 on hold-out")

    def predict_proba(self, features: OvertakeFeatures) -> float:
        raise NotImplementedError
