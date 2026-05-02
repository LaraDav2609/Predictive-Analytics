"""Conformal prediction wrapper — distribution-free calibration guarantees.

For trading, the most important property is: when the model says "70%", it should
be right ~70% of the time on out-of-sample races. Conformal prediction gives
this guarantee under exchangeability without distributional assumptions.

Usage:
    cal = ConformalCalibrator.fit(calib_probs, calib_outcomes, alpha=0.1)
    # For binary classification: prediction set {0}, {1}, or {0, 1}
    pred_set = cal.binary_prediction_set(new_prob)
    # For multi-class: include each class whose nonconformity ≤ threshold
    classes = cal.prediction_set(class_probs)

Method: split conformal — fit a base model on a training fold, compute non-
conformity scores on a held-out calibration fold, take the (1-α)-quantile as
the threshold. Marginal coverage ≥ 1-α holds in expectation under the
exchangeability assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConformalCalibrator:
    """Holds the threshold τ derived from calibration nonconformity scores.

    Score function for binary outcomes:
        score(prob, outcome) = 1 - prob if outcome == 1 else prob
        (low score = the model gave high probability to the realized class)

    Coverage guarantee: P(y ∈ prediction_set(x)) ≥ 1 - α (marginally).
    """
    threshold: float
    alpha: float = 0.1
    nonconformity_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def fit(
        cls,
        calib_probs: np.ndarray,
        calib_outcomes: np.ndarray,
        alpha: float = 0.1,
    ) -> "ConformalCalibrator":
        """Fit from calibration-set predictions and outcomes.

        `calib_probs` is either:
          - shape (n,) — binary; interpreted as P(class=1)
          - shape (n, K) — multi-class; column k = P(class=k)
        `calib_outcomes` is shape (n,) of class indices (0/1 for binary,
        0..K-1 for multi-class).
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        n = len(calib_outcomes)
        if n == 0:
            raise ValueError("calibration set is empty")

        scores = _nonconformity_scores(calib_probs, calib_outcomes)
        # Conformal quantile correction: ⌈(n+1)(1-α)⌉ / n
        q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        threshold = float(np.quantile(scores, q_level, method="higher"))
        return cls(threshold=threshold, alpha=alpha, nonconformity_scores=scores)

    def binary_prediction_set(self, prob: float) -> list[int]:
        """For binary classification with P(class=1) = prob.

        Includes class 1 if (1 - prob) ≤ threshold (i.e., prob ≥ 1 - threshold).
        Includes class 0 if prob ≤ threshold.
        Both ⇒ the model is "uncertain" by conformal definition; widen exposure.
        Neither ⇒ ill-formed (shouldn't happen with valid threshold).
        """
        out: list[int] = []
        if prob <= self.threshold:
            out.append(0)
        if (1 - prob) <= self.threshold:
            out.append(1)
        return out

    def prediction_set(self, class_probs: np.ndarray) -> list[int]:
        """For multi-class. Include class k if (1 - P(k)) ≤ threshold."""
        return [int(k) for k, p in enumerate(class_probs) if (1 - p) <= self.threshold]

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Identity. Conformal prediction is a *set-valued* output, not a
        probability transformer; use Platt or Isotonic calibration for
        probability adjustment, and use the prediction-set methods on this
        class for coverage-guaranteed outputs.
        """
        return np.asarray(probs)

    def empirical_coverage(self, probs: np.ndarray, outcomes: np.ndarray) -> float:
        """Sanity: fraction of test points whose true class falls in the
        prediction set. Should be ≥ 1 - α on held-out data."""
        outcomes = np.asarray(outcomes, dtype=int)
        probs = np.asarray(probs)
        in_set = 0
        if probs.ndim == 1:
            for p, y in zip(probs, outcomes):
                if y in self.binary_prediction_set(float(p)):
                    in_set += 1
        else:
            for row, y in zip(probs, outcomes):
                if y in self.prediction_set(row):
                    in_set += 1
        return in_set / len(outcomes) if len(outcomes) else 0.0


def _nonconformity_scores(calib_probs: np.ndarray, calib_outcomes: np.ndarray) -> np.ndarray:
    """Score = 1 - P(realized class). Lower means the model gave high
    confidence to the right answer."""
    probs = np.asarray(calib_probs, dtype=float)
    outcomes = np.asarray(calib_outcomes, dtype=int)
    if probs.ndim == 1:
        # Binary: P(realized) = prob if outcome=1 else 1-prob
        prob_of_truth = np.where(outcomes == 1, probs, 1.0 - probs)
    else:
        # Multi-class
        prob_of_truth = probs[np.arange(len(outcomes)), outcomes]
    return 1.0 - prob_of_truth
