"""Diffusion model for race-trajectory generation — generative counterfactuals.

Train a diffusion model on (lap, position, gap, tire_age) trajectories. Sample
from it to generate plausible alternative race outcomes that the physics-based
Monte Carlo can't produce because it's constrained to the simulator's
hard-coded dynamics.

Use cases:
  - Stress-test market predictions: what's the model's edge under realistic
    rare-event trajectories the MC misses?
  - Counterfactuals: condition on "Verstappen DNFs lap 5" and sample the rest
    of the race.

Lower priority — the physics MC is more interpretable and easier to debug.
Build only if calibration on tail markets (margin of victory, exact-finish-order)
needs improvement.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajectoryDiffusion(nn.Module):
    def __init__(self, n_drivers: int, n_laps: int, n_features: int, n_diffusion_steps: int = 1000) -> None:
        super().__init__()
        self.n_drivers = n_drivers
        self.n_laps = n_laps
        self.n_features = n_features
        self.n_steps = n_diffusion_steps
        # TODO: a U-Net-1D or transformer denoiser

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("standard DDPM denoiser predicting noise")

    def sample(self, condition: dict | None = None, n_samples: int = 1) -> torch.Tensor:
        """Conditional or unconditional sampling. Condition can pin events
        (driver X DNFs at lap K) for counterfactual analysis."""
        raise NotImplementedError
