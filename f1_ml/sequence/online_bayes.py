"""Online Bayesian updating during a live race.

Pre-race posterior over (pace, tire_deg) per driver becomes the prior at lap 1.
Each lap, observe a corrected lap time → conjugate update → new posterior →
re-run a fast MC to refresh market probabilities.

Critical for v2 (in-race trading): retraining the full hierarchical model lap-by-lap
is too expensive; conjugate Gaussian updates take microseconds.

For non-conjugate models, fall back to particle filtering or SMC.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GaussianBelief:
    mean: float
    variance: float


def conjugate_update(
    prior: GaussianBelief,
    observation: float,
    obs_variance: float,
) -> GaussianBelief:
    """Closed-form Gaussian conjugate update — O(1) per lap per driver."""
    new_var = 1.0 / (1.0 / prior.variance + 1.0 / obs_variance)
    new_mean = new_var * (prior.mean / prior.variance + observation / obs_variance)
    return GaussianBelief(new_mean, new_var)


class ParticleFilter:
    """For non-Gaussian posteriors (e.g., bimodal under uncertain weather)."""

    def __init__(self, n_particles: int = 1000) -> None:
        self.n_particles = n_particles

    def step(self, observation: float) -> None:
        raise NotImplementedError("propose, weight, resample")
