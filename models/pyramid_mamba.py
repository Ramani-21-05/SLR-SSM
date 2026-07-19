"""
Hierarchical Multi-Scale Mamba decoder.

This is the module that must NOT read as "STNet's MTPM but with Mamba
blocks instead of conv kernels." MTPM runs several parallel 1D convs of
different kernel sizes on the SAME-length sequence. This module instead
builds a genuine temporal pyramid: each level operates on the sequence at a
different time resolution, and the levels are merged back together.

Level 0: full resolution   (T)    -- local, frame-to-frame detail
Level 1: downsampled by 2x (T/2)  -- short-range gesture segments
Level 2: downsampled by 4x (T/4)  -- longer-range sign/word-level context

Each level is its own small Mamba stack. Coarser levels are upsampled back
to T and summed with the fine level, then passed through a residual
connection -- analogous in spirit to a U-Net's multi-resolution merge,
not to MTPM's multi-kernel-size parallel convs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba import MambaBlock


class MambaStack(nn.Module):
    """A small stack of Mamba blocks at one temporal resolution."""

    def __init__(self, d_model: int, depth: int = 2, d_state: int = 16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class HierarchicalMultiScaleMamba(nn.Module):
    def __init__(self, d_model: int = 512, depth_per_level: int = 2, d_state: int = 16):
        super().__init__()
        self.d_model = d_model

        self.level0 = MambaStack(d_model, depth=depth_per_level, d_state=d_state)
        self.level1 = MambaStack(d_model, depth=depth_per_level, d_state=d_state)
        self.level2 = MambaStack(d_model, depth=depth_per_level, d_state=d_state)

        self.downsample1 = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)
        self.downsample2 = nn.AvgPool1d(kernel_size=4, stride=4, ceil_mode=True)

        self.merge = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)"""
        B, T, D = x.shape

        # Level 0: full resolution.
        y0 = self.level0(x)

        # Level 1: half resolution.
        x1 = self.downsample1(x.transpose(1, 2)).transpose(1, 2)  # (B, ~T/2, D)
        y1 = self.level1(x1)
        y1 = F.interpolate(y1.transpose(1, 2), size=T, mode="linear", align_corners=False)
        y1 = y1.transpose(1, 2)  # back to (B, T, D)

        # Level 2: quarter resolution.
        x2 = self.downsample2(x.transpose(1, 2)).transpose(1, 2)  # (B, ~T/4, D)
        y2 = self.level2(x2)
        y2 = F.interpolate(y2.transpose(1, 2), size=T, mode="linear", align_corners=False)
        y2 = y2.transpose(1, 2)

        merged = self.merge(y0 + y1 + y2) + x  # residual from the original input
        return merged
