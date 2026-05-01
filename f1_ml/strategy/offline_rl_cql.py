"""Conservative Q-Learning (CQL) for offline pit strategy learning.

Why offline: F1 has ~22 races/yr × 20 cars = ~440 episodes/yr. We can't run
live experiments. CQL is the SOTA offline RL algorithm — penalizes Q-values
on out-of-distribution actions to prevent the policy from exploiting
underexplored regions.

State: race state vector (positions, gaps, tire ages, compounds, lap, weather).
Action: discrete (stay_out / pit_soft / pit_med / pit_hard).
Reward: -lap_time at each step; +finish_position-bonus at race end.

Trained from historical race trajectories → deployed as a policy for the
simulator's strategy module (replacing dp_optimal_stop.py for v3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CQLPolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))
        self.q2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)


def cql_loss(
    q_values: torch.Tensor,
    target_q: torch.Tensor,
    actions: torch.Tensor,
    cql_alpha: float = 1.0,
) -> torch.Tensor:
    """Standard CQL loss: TD error + alpha * (logsumexp Q over all actions - Q(s, a_data))."""
    raise NotImplementedError("delegate to d3rlpy CQL implementation")
