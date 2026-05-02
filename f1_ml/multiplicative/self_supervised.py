"""Self-supervised pretraining on unlabeled telemetry.

Labeled outcome data is scarce (~220 races/decade). Unlabeled telemetry is
huge — every practice and qualifying session adds millions of ticks. Pretrain
the Transformer / LSTM encoder with a masked-segment objective (mask 15% of
mini-sector samples, predict them from context), then fine-tune the
multi-task heads on the small labeled set.

Pattern: BERT-style for racing data. Empirically gives 5-15% improvement on
downstream tasks when labeled data is small.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedSegmentPretrainer(nn.Module):
    def __init__(self, encoder: nn.Module, n_features: int, mask_pct: float = 0.15) -> None:
        super().__init__()
        self.encoder = encoder
        self.recon_head = nn.Linear(getattr(encoder, "feature_dim", 128), n_features)
        self.mask_pct = mask_pct

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "randomly mask mask_pct of timesteps with a learned MASK embedding; "
            "encoder produces context; recon_head predicts the masked features"
        )


def contrastive_pretraining_objective(z_view_a: torch.Tensor, z_view_b: torch.Tensor) -> torch.Tensor:
    """SimCLR-style: two augmented views of the same lap segment should embed close,
    different segments should embed far. NT-Xent loss."""
    raise NotImplementedError
