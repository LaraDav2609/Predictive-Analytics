"""Computer vision tire-wear classifier — CNN on broadcast frames.

Goal: estimate tire wear visually (graining, blistering, exposed canvas) before
it shows up in lap-time degradation. Provides 1-3 lap warning that lets the
model anticipate a forced stop or a sudden DNF risk.

Data access is the gating issue: needs frame extraction from broadcast video
(rights / pipeline) plus labeled tire-condition examples (manual annotation
of a few hundred onboard / pit-stop frames).

Architecture: ResNet-50 / ConvNeXt backbone, multi-task head (wear_pct,
graining_present, blister_present). Pretrained on ImageNet; fine-tune on
labeled F1 frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TireWearPrediction:
    wear_pct: float
    graining_prob: float
    blister_prob: float


class TireWearCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: from torchvision.models import resnet50
        self.backbone: nn.Module | None = None
        self.head_wear = nn.Linear(2048, 1)
        self.head_graining = nn.Linear(2048, 1)
        self.head_blister = nn.Linear(2048, 1)

    def forward(self, image: torch.Tensor) -> TireWearPrediction:
        raise NotImplementedError
