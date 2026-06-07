"""Tests for sports.f1.ml.sequence.online_bayes — Gaussian conjugate + ParticleFilter."""

from __future__ import annotations

import numpy as np
import pytest

from sports.f1.ml.sequence.online_bayes import (
    GaussianBelief,
    ParticleFilter,
    conjugate_update,
)


# ------------------------------------------------------------ conjugate_update

def test_conjugate_update_shrinks_variance():
    prior = GaussianBelief(mean=85.0, variance=1.0)
    post = conjugate_update(prior, observation=84.5, obs_variance=0.5)
    assert post.variance < prior.variance


def test_conjugate_update_pulls_mean_toward_observation():
    prior = GaussianBelief(mean=85.0, variance=1.0)
    post = conjugate_update(prior, observation=80.0, obs_variance=1.0)
    assert prior.mean > post.mean > 80.0


def test_conjugate_update_high_obs_noise_keeps_prior_dominant():
    """Observation with huge variance should barely move the posterior."""
    prior = GaussianBelief(mean=85.0, variance=0.1)
    post = conjugate_update(prior, observation=80.0, obs_variance=1000.0)
    assert abs(post.mean - prior.mean) < 0.1


# -------------------------------------------------------------- ParticleFilter

def test_particle_filter_initialize_seeds_particles():
    pf = ParticleFilter(n_particles=500)
    pf.initialize(prior_mean=85.0, prior_sigma=1.0)
    assert pf.particles.shape == (500,)
    assert abs(pf.posterior_mean - 85.0) < 0.5


def test_particle_filter_step_pulls_mean_toward_observations():
    pf = ParticleFilter(n_particles=2000, transition_sigma=0.05, obs_sigma=0.3, seed=0)
    pf.initialize(prior_mean=85.0, prior_sigma=2.0)
    # Stream 30 observations centered at 80; posterior should follow.
    rng = np.random.default_rng(0)
    for _ in range(30):
        pf.step(observation=float(80.0 + rng.normal(0, 0.1)))
    assert abs(pf.posterior_mean - 80.0) < 0.5


def test_particle_filter_variance_shrinks_with_data():
    pf = ParticleFilter(n_particles=2000, transition_sigma=0.0, obs_sigma=0.3, seed=0)
    pf.initialize(prior_mean=85.0, prior_sigma=2.0)
    initial_var = pf.posterior_var
    for _ in range(20):
        pf.step(observation=80.0)
    assert pf.posterior_var < initial_var


def test_particle_filter_step_without_initialize_raises():
    pf = ParticleFilter()
    with pytest.raises(RuntimeError, match="not initialized"):
        pf.step(observation=80.0)


def test_particle_filter_rejects_invalid_init():
    with pytest.raises(ValueError):
        ParticleFilter(n_particles=0)
    with pytest.raises(ValueError):
        ParticleFilter(obs_sigma=0)


def test_particle_filter_initialize_rejects_bad_sigma():
    pf = ParticleFilter()
    with pytest.raises(ValueError):
        pf.initialize(prior_mean=85.0, prior_sigma=0)


def test_particle_filter_custom_log_likelihood_used():
    """Caller-provided likelihood lets the filter handle non-Gaussian observation models."""
    pf = ParticleFilter(n_particles=500, transition_sigma=0.0, obs_sigma=1.0, seed=0)
    pf.initialize(prior_mean=85.0, prior_sigma=2.0)

    # Sharp Laplace likelihood instead of Gaussian.
    def laplace_loglik(particles: np.ndarray, obs: float) -> np.ndarray:
        return -np.abs(particles - obs) / 0.1

    for _ in range(10):
        pf.step(observation=80.0, log_likelihood=laplace_loglik)
    assert abs(pf.posterior_mean - 80.0) < 1.0


def test_particle_filter_resamples_when_ess_low():
    """After many degenerate weights, weights should renormalize to ~uniform."""
    pf = ParticleFilter(n_particles=200, transition_sigma=0.0, obs_sigma=0.05, seed=0)
    pf.initialize(prior_mean=85.0, prior_sigma=5.0)
    pf.step(observation=80.0)
    # After resampling, weights should be close to uniform.
    assert pf.weights.std() < 0.01
