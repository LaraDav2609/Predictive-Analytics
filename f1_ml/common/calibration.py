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
    """Logistic post-hoc calibration. Fit on held-out validation predictions.

    Maps raw model probabilities `p` through a 1-feature logistic regression on
    the logit of `p`:  calibrated = sigmoid(a * logit(p) + b).

    `a=1, b=0` is the identity (no recalibration). Fit produces (a, b) that
    minimize log loss on held-out (probs, outcomes) pairs.
    """

    def __init__(self) -> None:
        self.a: float = 1.0
        self.b: float = 0.0

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "PlattScaler":
        from sklearn.linear_model import LogisticRegression

        eps = 1e-9
        p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
        logits = np.log(p / (1 - p))
        y = np.asarray(outcomes, dtype=int)
        if len(np.unique(y)) < 2:
            # Only one outcome class observed; cannot fit. Leave at identity.
            return self
        model = LogisticRegression(C=1e9, solver="lbfgs")  # near-unregularized
        model.fit(logits.reshape(-1, 1), y)
        self.a = float(model.coef_[0, 0])
        self.b = float(model.intercept_[0])
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-9
        p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
        logits = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-(self.a * logits + self.b)))


class IsotonicCalibrator:
    """Non-parametric monotonic calibration. More flexible than Platt; needs more data.

    Wraps sklearn.isotonic.IsotonicRegression. Output is a step-monotone
    function of the input probabilities; clipped to [0, 1] at the boundaries.
    """

    def __init__(self) -> None:
        self._model = None  # type: ignore[assignment]

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(np.asarray(probs, dtype=float), np.asarray(outcomes, dtype=float))
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._model is None:
            # Identity if not yet fit.
            return np.asarray(probs, dtype=float)
        return self._model.predict(np.asarray(probs, dtype=float))
