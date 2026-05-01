"""Inverse RL — recover what each team is *actually* optimizing.

Teams sometimes care more about championship points than this race's win
probability (sandbagging for grid penalty management, saving tires for next
weekend, defending P10 to lock in a single point). A pure race-time optimizer
mispredicts these strategies.

IRL learns a per-team reward function from observed strategy choices, then we
sample policies under that reward in the simulator.

Implementation: maximum entropy IRL (Ziebart 2008) or adversarial IRL (GAIL).
Lower priority — only worth doing once v2 ships and we see systematic strategy
mispredictions for specific teams.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TeamRewardFunction:
    team_code: str
    weights: dict[str, float]  # e.g., {"race_position": 1.0, "championship_points": 0.4, ...}


class MaxEntIRL:
    def __init__(self, feature_dim: int, lr: float = 0.01) -> None:
        self.reward_net = nn.Linear(feature_dim, 1)
        self.lr = lr

    def fit(self, expert_trajectories: list, simulator: object) -> TeamRewardFunction:
        raise NotImplementedError("MaxEnt IRL: match expert feature expectations")
