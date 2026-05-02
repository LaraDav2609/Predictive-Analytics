"""LSTM pace model — sequence-to-sequence next-lap pace prediction.

Input: sequence of (corrected lap time, tire age, compound, dirty air, weather,
position) for laps 1..N. Output: pace distribution for lap N+1.

Captures momentum effects (driver in a rhythm vs. struggling), fuel-burn
non-linearities, and warm-up patterns across compounds that the per-lap GBM
heads in core/gbm_pace.py miss.

Lighter / faster than the Transformer; train both, ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMPaceModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
        self.head_mean = nn.Linear(hidden, 1)
        self.head_log_sigma = nn.Linear(hidden, 1)  # heteroskedastic — uncertainty grows under SC etc.

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return self.head_mean(last).squeeze(-1), self.head_log_sigma(last).squeeze(-1)


def gaussian_nll_loss(mean: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Standard heteroskedastic NLL — model learns its own uncertainty."""
    sigma = torch.exp(log_sigma)
    return ((target - mean) ** 2 / (2 * sigma ** 2) + log_sigma).mean()
