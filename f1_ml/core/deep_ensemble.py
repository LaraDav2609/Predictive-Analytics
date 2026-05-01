"""Deep ensemble — train K independent neural nets with different inits;
ensemble-mean is the prediction, ensemble-variance is the uncertainty.

Cheaper than full Bayesian deep learning, comparable calibration quality. The
practical default for getting calibrated uncertainty out of a pace / overtake
neural net.

K = 5 is the sweet spot; more gives diminishing returns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn


class DeepEnsemble:
    def __init__(self, model_factory: Callable[[], nn.Module], k: int = 5) -> None:
        self.model_factory = model_factory
        self.k = k
        self.models: list[nn.Module] = []

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 50) -> "DeepEnsemble":
        raise NotImplementedError("train k models with independent random inits; same data, different seeds")

    def predict(self, X: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) over the ensemble."""
        raise NotImplementedError
