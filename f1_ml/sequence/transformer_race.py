"""Transformer race-state model — joint encoder over the full grid.

Input per lap: tensor [n_drivers, n_features] with positional encoding for both
lap number AND grid position. Self-attention across drivers captures
inter-driver effects (DRS trains, undercut threats) that per-driver sequence
models miss; cross-attention to lap history captures temporal dynamics.

Output heads (multi-task): next-lap pace, DNF hazard, overtake probability per
adjacent pair.

This is the v2 workhorse. Self-supervised pretraining (multiplicative/self_supervised.py)
gives it a head start on small labeled data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RaceTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        max_drivers: int = 22,
        max_laps: int = 80,
    ) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(n_features, d_model)
        self.driver_pos = nn.Embedding(max_drivers, d_model)
        self.lap_pos = nn.Embedding(max_laps, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.pace_head = nn.Linear(d_model, 1)
        self.dnf_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, driver_idx: torch.Tensor, lap_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "project features + add driver and lap positional embeddings; encode; multi-head outputs"
        )
