"""Conformal prediction wrapper — distribution-free calibration guarantees.

For trading, the most important property is: when the model says "70%", it should
be right ~70% of the time on out-of-sample races. Conformal prediction gives
this guarantee under exchangeability without distributional assumptions.

Usage: fit any base model, then `ConformalCalibrator.fit(base_model, calib_set)`
and `transform(probs)` to get conformalized probabilities.

Method: split conformal (mondrian / class-conditional for multi-class race winner).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalCalibrator:
    nonconformity_scores: np.ndarray
    alpha: float = 0.1  # 1 - target coverage

    @classmethod
    def fit(cls, calib_probs: np.ndarray, calib_outcomes: np.ndarray, alpha: float = 0.1) -> "ConformalCalibrator":
        raise NotImplementedError(
            "score(prob, outcome) = 1 - prob if outcome else prob; quantile of scores at level 1-alpha"
        )

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Adjusts probabilities to satisfy the marginal coverage guarantee."""
        raise NotImplementedError

    def prediction_set(self, class_probs: np.ndarray) -> list[int]:
        """For multi-class (e.g., race winner): returns set of classes with conformal coverage."""
        raise NotImplementedError
