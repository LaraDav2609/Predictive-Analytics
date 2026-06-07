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
    apply_sigmoid: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic forward passes with dropout active at inference.

    Returns (mean, std) over `n_samples` passes. Set `apply_sigmoid=True`
    when the model outputs raw classification logits.

    Note: this only produces non-zero variance if the model contains at least
    one Dropout layer. Pure deterministic models will return zero std.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    # Keep dropout layers in training mode while freezing batch norm.
    was_training = model.training
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            m.eval()

    with torch.no_grad():
        preds = []
        for _ in range(n_samples):
            out = model(X)
            if apply_sigmoid:
                out = torch.sigmoid(out)
            preds.append(out.detach().cpu().numpy())
    if was_training is False:
        model.eval()

    stack = np.stack(preds, axis=0)
    return np.asarray(stack.mean(axis=0)), np.asarray(stack.std(axis=0))
