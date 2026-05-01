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
    skills: np.ndarray  # shape (n_drivers,)


def fit(finish_orders: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-6) -> BradleyTerryFit:
    """MM (minorization-maximization) algorithm; converges quickly for BT."""
    raise NotImplementedError("standard MM iteration on pairwise win counts")


def predict_h2h(fit_: BradleyTerryFit, driver_a: str, driver_b: str) -> float:
    """P(A finishes ahead of B). For sanity-checking H2H markets."""
    raise NotImplementedError
