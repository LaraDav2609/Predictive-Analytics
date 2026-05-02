"""Bradley-Terry pairwise-comparison strength model.

P(i beats j) = exp(s_i) / (exp(s_i) + exp(s_j))

Fit via maximum likelihood on all observed head-to-head finish orderings. Gives
a single skill scalar per driver — the simplest ratings baseline.

Limitation: doesn't separate driver from car. Use as a sanity baseline only;
production should use hierarchical_bayes.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BradleyTerryFit:
    driver_codes: list[str]
    skills: np.ndarray  # shape (n_drivers,) — log-skills (so softmax-friendly)


def fit(
    finish_orders: pd.DataFrame,
    max_iter: int = 1000,
    tol: float = 1e-6,
    driver_col: str = "driver_code",
    race_col: str = "race",
    position_col: str = "position",
) -> BradleyTerryFit:
    """Minorization-Maximization for the Bradley-Terry model.

    For each race, every (winner, loser) pair where winner.position < loser.position
    contributes a head-to-head observation. The MM update rule is:

        skill_i ← (wins_i) / Σ_j  ((wins_ij + wins_ji) / (skill_i + skill_j))

    where wins_ij is i's wins over j summed across races. The algorithm is
    monotonic in the log-likelihood and converges to the global MLE because BT
    is concave in log-skills.

    Returns log-skills (i.e., the natural logarithm of multiplicative skills),
    so softmax(skills) gives win probabilities and predict_h2h is sigmoid of
    the difference.
    """
    if finish_orders.empty:
        return BradleyTerryFit(driver_codes=[], skills=np.array([]))

    drivers = sorted(finish_orders[driver_col].unique().tolist())
    n = len(drivers)
    idx_of = {d: i for i, d in enumerate(drivers)}

    # Build pairwise wins matrix W[i, j] = times i beat j.
    W = np.zeros((n, n), dtype=float)
    for _, race_df in finish_orders.groupby(race_col):
        ordered = race_df.sort_values(position_col)
        codes = ordered[driver_col].tolist()
        # i beats j if i finished ahead of j.
        for rank_i, code_i in enumerate(codes):
            i = idx_of[code_i]
            for code_j in codes[rank_i + 1:]:
                j = idx_of[code_j]
                W[i, j] += 1

    wins_total = W.sum(axis=1)  # row sums

    # Initialize multiplicative skills uniformly.
    p = np.ones(n)
    pair_count = W + W.T  # symmetric: total games between i and j

    for it in range(max_iter):
        new_p = np.zeros(n)
        for i in range(n):
            if wins_total[i] == 0:
                # Driver never won; assign tiny skill below the rest.
                new_p[i] = 1e-9
                continue
            denom = 0.0
            for j in range(n):
                if i == j or pair_count[i, j] == 0:
                    continue
                denom += pair_count[i, j] / (p[i] + p[j])
            new_p[i] = wins_total[i] / denom if denom > 0 else 1e-9
        # Normalize to keep numerics stable (BT is invariant to scale).
        new_p = new_p / new_p.sum() * n
        if np.max(np.abs(new_p - p)) < tol:
            p = new_p
            break
        p = new_p

    # Return log-skills — softmax-friendly.
    log_skills = np.log(np.maximum(p, 1e-12))
    log_skills -= log_skills.mean()  # center for interpretability
    return BradleyTerryFit(driver_codes=drivers, skills=log_skills)


def predict_h2h(fit_: BradleyTerryFit, driver_a: str, driver_b: str) -> float:
    """P(A finishes ahead of B) = sigmoid(skill_a - skill_b)."""
    if driver_a not in fit_.driver_codes or driver_b not in fit_.driver_codes:
        raise KeyError(f"unknown driver(s): {driver_a}, {driver_b}")
    a_idx = fit_.driver_codes.index(driver_a)
    b_idx = fit_.driver_codes.index(driver_b)
    diff = fit_.skills[a_idx] - fit_.skills[b_idx]
    # Clip for numerical stability before exp.
    diff = np.clip(diff, -30.0, 30.0)
    return float(1.0 / (1.0 + np.exp(-diff)))
