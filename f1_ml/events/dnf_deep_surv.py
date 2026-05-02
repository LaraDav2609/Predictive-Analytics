"""DeepSurv — neural-network hazard model. Cox PH with a learned non-linear
covariate transformation.

Beats the linear Cox model when feature interactions matter (e.g., high G's are
only catastrophic combined with high engine mileage). Needs more data than a
single season provides; train across multiple seasons.

Implementation: MLP outputs a single risk score; loss is the Cox partial likelihood.
Add MC dropout at inference for hazard uncertainty.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSurvNet(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def cox_partial_likelihood_loss(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Standard Cox partial likelihood, sorted by duration descending."""
    raise NotImplementedError("sort by duration desc; logsumexp over risk sets")
