"""Monte Carlo dropout — approximate Bayesian uncertainty for free.

Keep dropout active at inference, run T forward passes, treat the predictive
samples as draws from an approximate posterior. Simpler than deep ensembles
(one model instead of K) at the cost of slightly worse calibration.

Useful when training K ensemble members is too expensive (e.g., the LSTM /
Transformer pace models in sequence/).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mc_dropout_predict(
    model: nn.Module,
    X: torch.Tensor,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, std) over n_samples stochastic forward passes."""
    raise NotImplementedError(
        "model.train() to keep dropout active; loop n_samples passes; stack and reduce"
    )
