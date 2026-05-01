"""Plackett-Luce ranking model — the natural multi-way generalization of Bradley-Terry.

For a race with n drivers, P(observed finish order) factors as a chain of
softmax choices: P(winner) × P(2nd | winner removed) × ...

Fit by gradient descent on the negative log-likelihood of all historical race
orderings.

Useful as a baseline for full grid-order predictions; output skill scalars also
act as priors for the hierarchical model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PlackettLuceFit:
    driver_codes: list[str]
    skills: np.ndarray
    log_likelihood: float


def fit(finish_orders: pd.DataFrame, lr: float = 0.05, max_iter: int = 2000) -> PlackettLuceFit:
    raise NotImplementedError("autograd over PL log-likelihood; or closed-form MM updates")


def race_win_probabilities(fit_: PlackettLuceFit, entrants: list[str]) -> dict[str, float]:
    """P(winner | entrants); softmax over entrant skills."""
    raise NotImplementedError
