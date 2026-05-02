"""Kalman filter on latent "true pace" — each car's pace is a hidden state that
evolves smoothly; observed lap times are noisy realizations after subtracting
known offsets (fuel, tire age, dirty air, track evolution).

Output: per-driver pace estimate that updates each lap, with uncertainty.
Critical for the simulator: feeds the per-lap pace distribution.

State: scalar (or 2D with velocity) pace.
Observation: fuel- and dirty-air-corrected, tire-deg-adjusted lap time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PaceState:
    mean_s: float        # current best estimate of underlying pace
    variance_s2: float   # uncertainty


@dataclass
class KalmanConfig:
    process_noise_s2: float = 0.05  # how fast pace can drift lap-to-lap
    obs_noise_s2: float = 0.20      # noise in a single corrected lap time


class PaceKalman:
    def __init__(self, config: KalmanConfig) -> None:
        self.config = config
        self.state: PaceState | None = None

    def initialize(self, prior_mean: float, prior_var: float) -> None:
        self.state = PaceState(prior_mean, prior_var)

    def update(self, corrected_lap_time_s: float) -> PaceState:
        """Standard 1-D Kalman update. Returns the new posterior."""
        if self.state is None:
            raise RuntimeError("PaceKalman not initialized; call initialize(prior_mean, prior_var) first")

        # Predict step: state mean unchanged (random-walk model), variance grows by process noise.
        pred_mean = self.state.mean_s
        pred_var = self.state.variance_s2 + self.config.process_noise_s2

        # Update step: combine prediction with observation by inverse-variance weighting.
        kalman_gain = pred_var / (pred_var + self.config.obs_noise_s2)
        new_mean = pred_mean + kalman_gain * (corrected_lap_time_s - pred_mean)
        new_var = (1.0 - kalman_gain) * pred_var

        self.state = PaceState(mean_s=new_mean, variance_s2=new_var)
        return self.state
