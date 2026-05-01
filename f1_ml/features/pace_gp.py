"""Gaussian process regression for pace over a race — non-parametric alternative
to Kalman.

Useful when degradation is non-monotonic (e.g., undulating pace from variable
tire warm-up, or a stint that goes through a SC restart). Handles missing laps
gracefully and gives principled uncertainty bands.

Trade-off vs. Kalman: GP is O(n^3) per fit, Kalman is O(1) per update. For live
in-race use, prefer Kalman; for offline backtesting / analysis, GP is fine.
"""

from __future__ import annotations

import numpy as np


class PaceGP:
    def __init__(self, length_scale_laps: float = 5.0, noise_s: float = 0.2) -> None:
        self.length_scale = length_scale_laps
        self.noise = noise_s

    def fit(self, laps: np.ndarray, corrected_pace_s: np.ndarray) -> "PaceGP":
        raise NotImplementedError("RBF kernel; could swap to Matern-3/2 for less smooth")

    def predict(self, laps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) per query lap."""
        raise NotImplementedError
