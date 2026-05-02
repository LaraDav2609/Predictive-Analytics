"""Mixture-of-experts — route predictions to a specialized model per regime.

Regimes: dry / wet, low-overtake / high-overtake track, high-deg / low-deg
compound, safety-car-active / green-flag. A gating network learns to route
each prediction to the expert best suited to that regime.

Better than stacking when regime-specific patterns are sharp (e.g., wet races
play by entirely different rules — overtake model trained on dry races would
mispredict wet behavior).

Sparse MoE (top-1 or top-2) keeps inference cheap.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MixtureOfExperts(nn.Module):
    def __init__(self, n_features: int, n_experts: int, expert_hidden: int = 64) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(n_features, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, 1))
            for _ in range(n_experts)
        ])
        self.gate = nn.Linear(n_features, n_experts)

    def forward(self, x: torch.Tensor, top_k: int = 2) -> torch.Tensor:
        raise NotImplementedError(
            "softmax gate; select top-k experts; weighted sum of expert outputs"
        )
