"""
SignMamba: the full Semester 1 model, wiring together every block in
architecture diagram order:

  RGB frames ----> MobileNetV3 -----------\
  Left hand  ----> MLP  --------------------\
  Right hand ----> MLP  ---------------------> Simple fusion -> Hierarchical
  Pose       ----> MLP  --------------------/    (concat+linear)   Mamba -> CTC
"""

import torch
import torch.nn as nn

from models.mobilenet import RGBEncoder
from models.landmark_encoder import LandmarkBranch
from models.fusion import SimpleFusion
from models.pyramid_mamba import HierarchicalMultiScaleMamba
from models.ctc_head import CTCHead


class SignMamba(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        rgb_dim: int = 256,
        stream_dim: int = 64,
        d_model: int = 512,
        mamba_depth_per_level: int = 2,
        blank_id: int = 0,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.rgb_encoder = RGBEncoder(rgb_dim=rgb_dim, pretrained=pretrained_backbone)
        self.landmark_branch = LandmarkBranch(stream_dim=stream_dim)
        self.fusion = SimpleFusion(rgb_dim=rgb_dim, stream_dim=stream_dim, d_model=d_model)
        self.decoder = HierarchicalMultiScaleMamba(
            d_model=d_model, depth_per_level=mamba_depth_per_level
        )
        self.ctc_head = CTCHead(d_model=d_model, vocab_size=vocab_size, blank_id=blank_id)

    def forward(self, rgb_frames: torch.Tensor, landmarks: dict) -> torch.Tensor:
        """
        rgb_frames: (B, T, 3, 224, 224)
        landmarks:  dict, see LandmarkBranch docstring
        Returns:    log_probs (T, B, vocab_size), ready for CTC loss / decoding
        """
        rgb_feat = self.rgb_encoder(rgb_frames)              # (B, T, rgb_dim)
        landmark_feats = self.landmark_branch(landmarks)      # dict of (B, T, stream_dim)
        fused = self.fusion(rgb_feat, landmark_feats)          # (B, T, d_model)
        decoded = self.decoder(fused)                          # (B, T, d_model)
        log_probs = self.ctc_head(decoded)                     # (T, B, vocab_size)
        return log_probs

    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        return self.ctc_head.compute_loss(log_probs, targets, input_lengths, target_lengths)

    def decode(self, log_probs):
        return self.ctc_head.greedy_decode(log_probs)
