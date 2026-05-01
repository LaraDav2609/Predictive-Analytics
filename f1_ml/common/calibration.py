"""Probability calibration metrics & post-hoc calibrators.

For prediction-market trading, calibration matters more than raw accuracy: a model
that says 70% must be right ~70% of the time, or Kelly sizing destroys you.
"""

from __future__ import annotations

import numpy as np


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probability and 0/1 outcome.
    Lower is better; a constant base-rate predictor gives a useful baseline."""
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(probs: np.ndarray, outcomes: np.ndarray, eps: float = 1e-12) -> float:
    """Negative log-likelihood. Heavily penalizes confident wrong calls."""
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)))


def reliability_curve(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predictions, return (mean_predicted, mean_observed, bin_count) for plotting."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_pred = np.zeros(n_bins)
    mean_obs = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_pred[b] = probs[mask].mean()
            mean_obs[b] = outcomes[mask].mean()
            counts[b] = int(mask.sum())
    return mean_pred, mean_obs, counts


class PlattScaler:
    """Logistic post-hoc calibration. Fit on held-out validation predictions."""

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "PlattScaler":
        raise NotImplementedError("fit logistic regression of outcomes on logit(probs)")

    def transform(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IsotonicCalibrator:
    """Non-parametric monotonic calibration. More flexible than Platt; needs more data."""

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        raise NotImplementedError("wrap sklearn.isotonic.IsotonicRegression")

    def transform(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
