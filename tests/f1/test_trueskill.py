"""Tests for sports.f1.ml.ratings.trueskill_ratings."""

from __future__ import annotations

import math

import pytest

from sports.f1.ml.ratings.trueskill_ratings import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    SIGMA_MIN,
    TrueSkillRating,
    conservative_skill,
    update_after_race,
)


def _make_rating(code: str, mu: float = DEFAULT_MU, sigma: float = DEFAULT_SIGMA) -> TrueSkillRating:
    return TrueSkillRating(driver_code=code, mu=mu, sigma=sigma)


def test_trueskill_winner_mu_increases():
    pre = [_make_rating("A"), _make_rating("B")]
    post = update_after_race(pre, finish_order=["A", "B"])
    post_by_code = {r.driver_code: r for r in post}
    assert post_by_code["A"].mu > pre[0].mu
    assert post_by_code["B"].mu < pre[1].mu


def test_trueskill_uncertainty_decreases_after_informative_result():
    pre = [_make_rating("A"), _make_rating("B")]
    post = update_after_race(pre, finish_order=["A", "B"])
    post_by_code = {r.driver_code: r for r in post}
    # Sigma should shrink from a 50-50 prior after observing a result.
    assert post_by_code["A"].sigma < pre[0].sigma
    assert post_by_code["B"].sigma < pre[1].sigma


def test_trueskill_dominant_driver_skill_grows_over_many_races():
    pre = [_make_rating("A"), _make_rating("B"), _make_rating("C")]
    for _ in range(15):
        pre = update_after_race(pre, finish_order=["A", "B", "C"])
    by_code = {r.driver_code: r for r in pre}
    assert by_code["A"].mu > by_code["B"].mu > by_code["C"].mu


def test_trueskill_dnf_loses_to_finishers():
    pre = [_make_rating("A"), _make_rating("B"), _make_rating("C")]
    post = update_after_race(
        pre,
        finish_order=["A", "B", "C"],
        dnf_drivers={"C"},
    )
    by_code = {r.driver_code: r for r in post}
    # C ranked behind both A and B — skill should drop.
    assert by_code["C"].mu < pre[2].mu
    # A still gains skill.
    assert by_code["A"].mu > pre[0].mu


def test_trueskill_sigma_is_floored():
    pre = [_make_rating("A", sigma=0.6), _make_rating("B", sigma=0.6)]
    for _ in range(50):
        pre = update_after_race(pre, finish_order=["A", "B"])
    for r in pre:
        assert r.sigma >= SIGMA_MIN


def test_trueskill_unknown_driver_in_finish_order_skipped():
    pre = [_make_rating("A"), _make_rating("B")]
    post = update_after_race(pre, finish_order=["A", "B", "ROOKIE"])
    codes = {r.driver_code for r in post}
    assert codes == {"A", "B"}


def test_trueskill_conservative_skill_subtracts_uncertainty():
    r = _make_rating("A", mu=30.0, sigma=2.0)
    assert conservative_skill(r, k=3.0) == pytest.approx(30.0 - 6.0)


def test_trueskill_dynamics_inflates_sigma_each_race():
    """Without a result, sigma should grow due to the tau dynamics term."""
    pre = [_make_rating("A", sigma=2.0), _make_rating("B", sigma=2.0)]
    # The dynamics drift is small per race but observable over many.
    post = update_after_race(pre, finish_order=["A"])  # only one driver — no pairwise update
    # The remaining driver's sigma should not have shrunk (no pairing to inform it),
    # but should have inflated by sqrt(σ² + τ²) and stayed ≥ SIGMA_MIN.
    expected = math.sqrt(2.0 ** 2 + (DEFAULT_SIGMA / 100) ** 2)
    assert post[0].sigma == pytest.approx(expected, rel=1e-6)


def test_trueskill_returns_new_objects():
    """Update should not mutate the pre-race ratings."""
    a = _make_rating("A")
    b = _make_rating("B")
    a_mu_before = a.mu
    b_mu_before = b.mu
    update_after_race([a, b], finish_order=["A", "B"])
    # Originals untouched.
    assert a.mu == a_mu_before
    assert b.mu == b_mu_before
