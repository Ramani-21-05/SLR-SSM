"""
RGB branch encoder.

Replaces STNet's ResNet-18 backbone with MobileNetV3-Small to cut parameter
count and enable edge deployment (Sem 1 goal: solve STNet's stated GPU
memory / batch-size-2 limitation).

Input:  (B, T, 3, 224, 224) RGB frame sequence
Output: (B, T, rgb_dim) per-frame feature vectors
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class RGBEncoder(nn.Module):
    def __init__(self, rgb_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tvm.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = tvm.mobilenet_v3_small(weights=weights)

        # Keep only the conv feature extractor, drop the ImageNet classifier head.
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MobileNetV3-Small's last conv block outputs 576 channels.
        backbone_out_dim = 576
        self.project = nn.Linear(backbone_out_dim, rgb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3, H, W) -> fold time into batch for the 2D CNN
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feat = self.features(x)            # (B*T, 576, h, w)
        feat = self.pool(feat).flatten(1)  # (B*T, 576)
        feat = self.project(feat)          # (B*T, rgb_dim)

        return feat.view(B, T, -1)         # (B, T, rgb_dim)
