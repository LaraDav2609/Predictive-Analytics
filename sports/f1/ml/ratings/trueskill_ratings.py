"""TrueSkill — Bayesian skill rating with explicit per-driver uncertainty.

Each driver carries (mu, sigma); after each race, both update via factor-graph
message passing. Naturally handles partial information (DNFs, mixed-quality fields)
and gives confidence intervals out of the box.

Better than ELO for sparse data (~20-22 races/yr). Worse than hierarchical Bayes
because it still doesn't formally separate driver from car.

Implementation: native pairwise update over consecutive finishers (Herbrich 2006
factor-graph reduces to the standard Gaussian conjugate update for the 2-player
case; for n>2 we apply that pairwise to adjacent pairs in the finish order). DNFs
are treated as tied amongst themselves and ranked behind every finisher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Default TrueSkill hyperparameters (Herbrich 2006).
DEFAULT_MU = 25.0
DEFAULT_SIGMA = 25.0 / 3.0       # so 99% interval ≈ [0, 50]
DEFAULT_BETA = DEFAULT_SIGMA / 2  # per-game skill noise
DEFAULT_TAU = DEFAULT_SIGMA / 100  # dynamics — small drift between games
SIGMA_MIN = 0.5                    # never collapse uncertainty completely


@dataclass
class TrueSkillRating:
    driver_code: str
    mu: float = DEFAULT_MU
    sigma: float = DEFAULT_SIGMA


def update_after_race(
    pre_race: list[TrueSkillRating],
    finish_order: list[str],
    dnf_drivers: set[str] | None = None,
    beta: float = DEFAULT_BETA,
    tau: float = DEFAULT_TAU,
) -> list[TrueSkillRating]:
    """Pairwise Bayesian update over the partial ordering.

    Adjacent pairs (winner, loser) in `finish_order` contribute one TrueSkill
    win/loss update each. `dnf_drivers` are appended to the end (tied amongst
    themselves) so finishers strictly outrank DNFs.

    Returns a NEW list of ratings; pre-race list is not mutated.
    """
    if dnf_drivers is None:
        dnf_drivers = set()

    rating_by_code = {r.driver_code: TrueSkillRating(r.driver_code, r.mu, r.sigma)
                      for r in pre_race}

    # Apply small dynamics drift before the race (uncertainty grows between events).
    for r in rating_by_code.values():
        r.sigma = math.sqrt(r.sigma ** 2 + tau ** 2)

    # Build a unique full ordering: finishers first (in finish_order), then DNFs.
    # Skip drivers we have no rating for (e.g., debutants present in the race
    # but not yet in pre_race).
    ordered = [d for d in finish_order if d in rating_by_code and d not in dnf_drivers]
    dnf_list = [d for d in finish_order if d in dnf_drivers and d in rating_by_code]
    # DNFs that didn't make it into finish_order at all (rare) get appended last.
    ordered_full = ordered + dnf_list

    # Pairwise update on adjacent pairs of finishers.
    for k in range(len(ordered) - 1):
        winner = rating_by_code[ordered[k]]
        loser = rating_by_code[ordered[k + 1]]
        _two_player_update(winner, loser, beta=beta)

    # Each DNF lost to the last finisher (and is tied with other DNFs).
    if ordered and dnf_list:
        last_finisher = rating_by_code[ordered[-1]]
        for d in dnf_list:
            _two_player_update(last_finisher, rating_by_code[d], beta=beta)

    # Floor sigma so the model can keep learning even from a long-stable driver.
    for r in rating_by_code.values():
        r.sigma = max(r.sigma, SIGMA_MIN)

    return [rating_by_code[d] for d in (d for d in finish_order if d in rating_by_code)]


def _two_player_update(winner: TrueSkillRating, loser: TrueSkillRating, beta: float) -> None:
    """In-place TrueSkill update for one 2-player game."""
    c2 = 2.0 * beta * beta + winner.sigma ** 2 + loser.sigma ** 2
    c = math.sqrt(c2)
    t = (winner.mu - loser.mu) / c

    v = _v(t)
    w = _w(t, v)

    winner.mu = winner.mu + (winner.sigma ** 2 / c) * v
    loser.mu = loser.mu - (loser.sigma ** 2 / c) * v

    winner_var_new = winner.sigma ** 2 * max(0.0, 1.0 - (winner.sigma ** 2 / c2) * w)
    loser_var_new = loser.sigma ** 2 * max(0.0, 1.0 - (loser.sigma ** 2 / c2) * w)
    winner.sigma = math.sqrt(max(winner_var_new, SIGMA_MIN ** 2))
    loser.sigma = math.sqrt(max(loser_var_new, SIGMA_MIN ** 2))


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _v(t: float) -> float:
    """v(t) = φ(t) / Φ(t) — TrueSkill 'mean update' factor."""
    cdf = _normal_cdf(t)
    if cdf < 1e-15:
        # Limit: as t → -∞, v(t) → -t (asymptotic identity to avoid div-by-zero).
        return -t
    return _normal_pdf(t) / cdf


def _w(t: float, v_val: float | None = None) -> float:
    """w(t) = v(t) * (v(t) + t) — TrueSkill 'variance update' factor; in [0,1]."""
    if v_val is None:
        v_val = _v(t)
    out = v_val * (v_val + t)
    return float(min(1.0, max(0.0, out)))


def conservative_skill(rating: TrueSkillRating, k: float = 3.0) -> float:
    """mu - k*sigma — TrueSkill's standard conservative-estimate formula."""
    return rating.mu - k * rating.sigma
