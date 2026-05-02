"""Multi-task learning — shared backbone, separate heads for pace / DNF / overtake / pit time.

In F1, these tasks are correlated (struggling driver is slower AND more likely
to DNF AND less likely to overtake). A shared encoder learns representations
useful for all of them; task-specific heads adapt the representation.

Implementation: shared MLP / Transformer backbone; per-task heads with task-
specific losses summed (with learnable weights via uncertainty weighting,
Kendall et al. 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedBackbone(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.feature_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MultiTaskModel(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.backbone = SharedBackbone(n_features)
        d = self.backbone.feature_dim
        self.pace_head = nn.Linear(d, 2)       # mean + log-sigma
        self.dnf_head = nn.Linear(d, 1)        # logit
        self.overtake_head = nn.Linear(d, 1)   # logit
        self.pit_time_head = nn.Linear(d, 2)   # mean + log-sigma
        # Learnable task-loss weights (uncertainty weighting)
        self.log_var = nn.Parameter(torch.zeros(4))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.backbone(x)
        return {
            "pace": self.pace_head(z),
            "dnf": self.dnf_head(z).squeeze(-1),
            "overtake": self.overtake_head(z).squeeze(-1),
            "pit_time": self.pit_time_head(z),
        }


def uncertainty_weighted_loss(losses: dict[str, torch.Tensor], log_vars: torch.Tensor) -> torch.Tensor:
    """Kendall-et-al combined loss: each task's loss is divided by exp(log_var)
    plus a regularizer; learnable per-task weights."""
    raise NotImplementedError
