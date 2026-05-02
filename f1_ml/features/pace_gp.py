"""Gaussian process regression for pace over a race — non-parametric alternative
to Kalman.

Useful when degradation is non-monotonic (e.g., undulating pace from variable
tire warm-up, or a stint that goes through a SC restart). Handles missing laps
gracefully and gives principled uncertainty bands.

Trade-off vs. Kalman: GP is O(n^3) per fit, Kalman is O(1) per update. For live
in-race use, prefer Kalman; for offline backtesting / analysis, GP is fine.

Implementation note: a hand-rolled RBF GP is fine for the ~50-100 lap horizon
of a single stint without the scikit-learn dependency overhead, and avoids
hyperparameter optimization issues that bite on noisy data.
"""

from __future__ import annotations

import numpy as np


class PaceGP:
    """Gaussian process with RBF kernel.

    K(x, x') = σ_f² · exp(-(x - x')² / (2 · ℓ²))

    The signal variance σ_f² is fit from the data range; the length scale ℓ
    and noise σ_n are user-chosen (defaults work well for one-stint pace data).
    """

    def __init__(self, length_scale_laps: float = 5.0, noise_s: float = 0.2) -> None:
        if length_scale_laps <= 0:
            raise ValueError("length_scale_laps must be > 0")
        if noise_s < 0:
            raise ValueError("noise_s must be >= 0")
        self.length_scale = float(length_scale_laps)
        self.noise = float(noise_s)
        self._train_x: np.ndarray | None = None
        self._train_y: np.ndarray | None = None
        self._signal_var: float = 1.0
        self._mean: float = 0.0
        self._L: np.ndarray | None = None  # Cholesky of (K + σ²I)
        self._alpha: np.ndarray | None = None  # L^-T L^-1 (y - mean)

    def _rbf(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Pairwise RBF kernel."""
        d2 = (x1[:, None] - x2[None, :]) ** 2
        return self._signal_var * np.exp(-d2 / (2.0 * self.length_scale ** 2))

    def fit(self, laps: np.ndarray, corrected_pace_s: np.ndarray) -> "PaceGP":
        x = np.asarray(laps, dtype=float).ravel()
        y = np.asarray(corrected_pace_s, dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError("laps and corrected_pace_s must have the same shape")
        if len(x) == 0:
            raise ValueError("at least one observation is required")

        self._mean = float(np.mean(y))
        residual = y - self._mean
        # Signal variance: sample variance of residuals (rough estimate, but
        # avoids the cost of marginal-likelihood optimization).
        self._signal_var = float(max(np.var(residual), 1e-3))

        K = self._rbf(x, x) + (self.noise ** 2) * np.eye(len(x))
        # Cholesky for numerically stable solve.
        self._L = np.linalg.cholesky(K + 1e-9 * np.eye(len(x)))
        self._alpha = np.linalg.solve(
            self._L.T, np.linalg.solve(self._L, residual)
        )
        self._train_x = x
        self._train_y = y
        return self

    def predict(self, laps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) per query lap."""
        if self._train_x is None or self._L is None or self._alpha is None:
            raise RuntimeError("PaceGP is not fitted; call .fit() first")

        x_star = np.asarray(laps, dtype=float).ravel()
        K_star = self._rbf(x_star, self._train_x)  # (n_star, n_train)
        mean = K_star @ self._alpha + self._mean

        v = np.linalg.solve(self._L, K_star.T)
        K_starstar_diag = np.full(len(x_star), self._signal_var)
        var = K_starstar_diag - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-9)
        return mean, np.sqrt(var)
