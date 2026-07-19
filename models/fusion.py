"""
Semester 1 fusion: deliberately simple.

Concatenate RGB features with all three landmark streams, project down to
d_model with one linear layer. No attention, no gating -- that's Semester
2's job (confidence-aware / attention fusion). Keeping Sem 1 simple isolates
whether the backbone + decoder changes alone improve on STNet, before a
more complex fusion mechanism is layered on top.

Swap point for Semester 2: replace this module's forward() with a
confidence-aware or cross-attention version that reads each stream's
presence flag (see landmark_encoder.py) to down-weight occluded hands.
Everything upstream (encoders) and downstream (decoder) stays unchanged.
"""

import torch
import torch.nn as nn


class SimpleFusion(nn.Module):
    def __init__(self, rgb_dim: int = 256, stream_dim: int = 64, d_model: int = 512):
        super().__init__()
        in_dim = rgb_dim + 3 * stream_dim  # rgb + left_hand + right_hand + pose
        self.project = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

    def forward(self, rgb_feat: torch.Tensor, landmark_feats: dict) -> torch.Tensor:
        """
        rgb_feat:       (B, T, rgb_dim)
        landmark_feats: dict of (B, T, stream_dim) from LandmarkBranch
        Returns:        (B, T, d_model)
        """
        fused = torch.cat(
            [
                rgb_feat,
                landmark_feats["left_hand"],
                landmark_feats["right_hand"],
                landmark_feats["pose"],
            ],
            dim=-1,
        )
        return self.project(fused)
