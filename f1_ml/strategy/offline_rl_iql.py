"""Implicit Q-Learning (IQL) — alternative offline RL for pit strategy.

IQL avoids the OOD-action problem by never querying Q-values on actions outside
the data: it uses expectile regression on the value function and an advantage-
weighted policy update. Often more stable than CQL on small datasets.

Use when CQL is over-conservative or the action space is large.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IQLAgent:
    def __init__(self, state_dim: int, n_actions: int, expectile: float = 0.7, beta: float = 3.0) -> None:
        self.q_net: nn.Module | None = None
        self.v_net: nn.Module | None = None
        self.policy: nn.Module | None = None
        self.expectile = expectile
        self.beta = beta

    def fit(self, transitions: list) -> None:
        raise NotImplementedError(
            "expectile loss for V; standard TD for Q; advantage-weighted regression for policy"
        )

    def act(self, state: torch.Tensor) -> int:
        raise NotImplementedError
