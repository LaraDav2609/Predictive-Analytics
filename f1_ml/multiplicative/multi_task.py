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


TASK_NAMES = ("pace", "dnf", "overtake", "pit_time")


def uncertainty_weighted_loss(
    losses: dict[str, torch.Tensor],
    log_vars: torch.Tensor,
    task_names: tuple[str, ...] = TASK_NAMES,
) -> torch.Tensor:
    """Kendall et al. 2018 multi-task loss with learnable per-task uncertainty.

    L_total = Σ_i  exp(-log_var_i) * L_i  +  log_var_i

    where `log_var_i = log(σ_i²)` is a learnable parameter. The first term
    down-weights losses with high task noise; the second is a regularizer that
    keeps log_var from drifting to -∞ (which would zero the loss).

    Args:
        losses: dict[task_name, scalar tensor] — each task's per-batch loss.
        log_vars: 1-D tensor with one entry per task, in `task_names` order.
        task_names: order of `log_vars`. Defaults to TASK_NAMES.

    Returns:
        Combined scalar loss.
    """
    if log_vars.shape != (len(task_names),):
        raise ValueError(
            f"log_vars shape {tuple(log_vars.shape)} != ({len(task_names)},)"
        )
    total = torch.zeros(1, dtype=log_vars.dtype, device=log_vars.device).squeeze()
    for i, name in enumerate(task_names):
        if name not in losses:
            continue
        precision = torch.exp(-log_vars[i])
        total = total + precision * losses[name] + log_vars[i]
    return total
