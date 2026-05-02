"""Plackett-Luce ranking model — the natural multi-way generalization of Bradley-Terry.

For a race with n drivers, P(observed finish order) factors as a chain of
softmax choices: P(winner) × P(2nd | winner removed) × ...

Fit via the MM algorithm (Hunter 2004): closed-form, monotonic, converges to
the global MLE because PL is concave in log-skills.

Useful as a baseline for full grid-order predictions; output skill scalars
also act as priors for the hierarchical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PlackettLuceFit:
    driver_codes: list[str]
    skills: np.ndarray  # log-skills (softmax → win probability)
    log_likelihood: float


def fit(
    finish_orders: pd.DataFrame,
    max_iter: int = 2000,
    tol: float = 1e-6,
    driver_col: str = "driver_code",
    race_col: str = "race",
    position_col: str = "position",
) -> PlackettLuceFit:
    """MM iteration for Plackett-Luce.

    For each race observed as ordering (d_1, d_2, …, d_K) with d_1 winning:
      L = Π_{k=1..K-1}  skill_{d_k} / Σ_{j ≥ k}  skill_{d_j}

    The MM update credits each driver who was either chosen at position k OR
    still in the remaining pool at position k:

        skill_i ← chosen_i / Σ_{race, k}  1[i in remaining at k] / pool_sum_at_k

    Returns log-skills so they're softmax-friendly.
    """
    if finish_orders.empty:
        return PlackettLuceFit(driver_codes=[], skills=np.array([]), log_likelihood=0.0)

    drivers = sorted(finish_orders[driver_col].unique().tolist())
    n = len(drivers)
    idx_of = {d: i for i, d in enumerate(drivers)}

    orderings: list[list[int]] = []
    for _, race_df in finish_orders.groupby(race_col):
        ordered = race_df.sort_values(position_col)
        orderings.append([idx_of[c] for c in ordered[driver_col].tolist()])

    # Numerator: number of times each driver was the "chosen" one across all
    # (race, position) where there was a real choice (>1 driver remaining).
    chosen = np.zeros(n)
    for ordering in orderings:
        for k in range(len(ordering) - 1):
            chosen[ordering[k]] += 1

    if chosen.sum() == 0:
        return PlackettLuceFit(driver_codes=drivers, skills=np.zeros(n), log_likelihood=0.0)

    p = np.ones(n)

    for it in range(max_iter):
        denom = np.zeros(n)
        for ordering in orderings:
            remaining = ordering.copy()
            while len(remaining) > 1:
                pool_sum = float(p[remaining].sum())
                inv_sum = 1.0 / pool_sum if pool_sum > 0 else 0.0
                for r in remaining:
                    denom[r] += inv_sum
                remaining.pop(0)

        new_p = np.where(denom > 0, chosen / denom, 1e-12)
        new_p = new_p / new_p.sum() * n  # rescale (PL invariant to scale)
        if np.max(np.abs(new_p - p)) < tol:
            p = new_p
            break
        p = new_p

    log_skills = np.log(np.maximum(p, 1e-12))
    log_skills -= log_skills.mean()  # center for interpretability

    # Diagnostic log-likelihood under the fitted log-skills (logsumexp denom).
    ll = 0.0
    for ordering in orderings:
        remaining = ordering.copy()
        while len(remaining) > 1:
            num = log_skills[remaining[0]]
            pool_logs = log_skills[remaining]
            max_log = float(pool_logs.max())
            denom_log = max_log + float(np.log(np.sum(np.exp(pool_logs - max_log))))
            ll += float(num - denom_log)
            remaining.pop(0)

    return PlackettLuceFit(driver_codes=drivers, skills=log_skills, log_likelihood=ll)


def race_win_probabilities(fit_: PlackettLuceFit, entrants: list[str]) -> dict[str, float]:
    """P(winner | entrants); softmax over entrant log-skills.

    Drivers in `entrants` not present in the fit are dropped silently — useful
    when a fitted model is queried for a race with rookies.
    """
    indices = [fit_.driver_codes.index(d) for d in entrants if d in fit_.driver_codes]
    if not indices:
        return {}
    log_skills = fit_.skills[indices]
    max_log = float(log_skills.max())
    expd = np.exp(log_skills - max_log)
    probs = expd / expd.sum()
    return {fit_.driver_codes[i]: float(p) for i, p in zip(indices, probs)}
