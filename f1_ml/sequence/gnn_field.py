"""Graph neural network over the on-track field — drivers as nodes,
on-track proximity as edges.

A pure per-driver model can't represent "Verstappen is in a DRS train behind
Norris and a slow Stroll" — but a GNN that propagates messages along proximity
edges captures exactly this. Especially useful for overtake / position-change
prediction.

Edge weights: 1 / gap_seconds (capped). Node features: pace, tire age, compound,
DRS-available. Layer-wise: GAT or GraphSAGE; 2-3 layers is plenty (race graph is small).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FieldGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        # TODO: from torch_geometric.nn import GATConv, global_mean_pool
        self.n_features = n_features
        self.hidden = hidden
        self.n_layers = n_layers

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "stack of GATConv layers; node-level outputs feed into pace / overtake heads"
        )


def build_proximity_graph(positions: list[float], gap_threshold_s: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """positions = list of cumulative race time per driver in finish order.
    Returns (edge_index [2, n_edges], edge_weight)."""
    raise NotImplementedError
