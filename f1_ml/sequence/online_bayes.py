"""Online Bayesian updating during a live race.

Pre-race posterior over (pace, tire_deg) per driver becomes the prior at lap 1.
Each lap, observe a corrected lap time → conjugate update → new posterior →
re-run a fast MC to refresh market probabilities.

Critical for v2 (in-race trading): retraining the full hierarchical model lap-by-lap
is too expensive; conjugate Gaussian updates take microseconds.

For non-conjugate models, fall back to particle filtering or SMC.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


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
    """Bootstrap particle filter for non-Gaussian latent state (e.g., bimodal
    pace under uncertain weather).

    State: a 1-D scalar (typically clean-air dry pace per driver).
    Transition: random walk with stddev `transition_sigma` per step.
    Likelihood: caller-provided log-likelihood function `log_likelihood(particle, obs)`,
    defaulting to a Gaussian centered at the particle.

    Each `step(obs)` performs:
      1. Propose: x_t' = x_{t-1} + N(0, transition_sigma²)
      2. Weight: w_t ∝ likelihood(obs | x_t')
      3. Resample: systematic resampling if effective sample size < n_particles/2
    """

    def __init__(
        self,
        n_particles: int = 1000,
        transition_sigma: float = 0.1,
        obs_sigma: float = 0.3,
        seed: int = 0,
    ) -> None:
        if n_particles < 1:
            raise ValueError("n_particles must be >= 1")
        if transition_sigma < 0 or obs_sigma <= 0:
            raise ValueError("sigmas must be positive (obs_sigma > 0)")
        self.n_particles = int(n_particles)
        self.transition_sigma = float(transition_sigma)
        self.obs_sigma = float(obs_sigma)
        self._rng = np.random.default_rng(seed)
        self.particles: np.ndarray = np.array([])
        self.weights: np.ndarray = np.array([])

    def initialize(self, prior_mean: float, prior_sigma: float) -> None:
        """Seed particles from a Gaussian prior. Call before .step()."""
        if prior_sigma <= 0:
            raise ValueError("prior_sigma must be > 0")
        self.particles = self._rng.normal(prior_mean, prior_sigma, size=self.n_particles)
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    def step(
        self,
        observation: float,
        log_likelihood: Callable[[np.ndarray, float], np.ndarray] | None = None,
    ) -> None:
        """Propagate, weight, and (if needed) resample.

        `log_likelihood(particles, obs)` returns shape (n_particles,) log-densities.
        If None, defaults to Gaussian log-pdf with `self.obs_sigma`.
        """
        if self.particles.size == 0:
            raise RuntimeError("ParticleFilter not initialized; call .initialize(...) first")

        # 1. Propose.
        self.particles = self.particles + self._rng.normal(
            0.0, self.transition_sigma, size=self.n_particles
        )

        # 2. Weight.
        if log_likelihood is None:
            log_w = -0.5 * ((self.particles - observation) / self.obs_sigma) ** 2
        else:
            log_w = log_likelihood(self.particles, observation)
        log_w -= log_w.max()  # numerical stability
        w = np.exp(log_w) * self.weights
        w_sum = float(w.sum())
        if w_sum <= 0:
            # Pathological: re-uniformize.
            w = np.full(self.n_particles, 1.0 / self.n_particles)
        else:
            w = w / w_sum
        self.weights = w

        # 3. Resample if ESS too low.
        ess = 1.0 / float(np.sum(w * w))
        if ess < self.n_particles / 2.0:
            self._systematic_resample()

    def _systematic_resample(self) -> None:
        positions = (self._rng.uniform() + np.arange(self.n_particles)) / self.n_particles
        cumulative = np.cumsum(self.weights)
        idx = np.searchsorted(cumulative, positions)
        idx = np.clip(idx, 0, self.n_particles - 1)
        self.particles = self.particles[idx]
        self.weights = np.full(self.n_particles, 1.0 / self.n_particles)

    @property
    def posterior_mean(self) -> float:
        if self.particles.size == 0:
            raise RuntimeError("ParticleFilter not initialized")
        return float(np.sum(self.particles * self.weights))

    @property
    def posterior_var(self) -> float:
        if self.particles.size == 0:
            raise RuntimeError("ParticleFilter not initialized")
        mean = self.posterior_mean
        return float(np.sum(self.weights * (self.particles - mean) ** 2))
