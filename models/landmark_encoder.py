"""
MediaPipe landmark branch encoders.

Three SEPARATE streams (left hand, right hand, pose) rather than one merged
MLP. This is deliberate, not incidental: keeping the streams independent now
is what lets Semester 2's confidence-aware fusion gate each hand's
contribution individually later, without restructuring this module.

Each stream also takes a presence flag (1.0 if MediaPipe detected the
landmarks that frame, 0.0 if not) concatenated onto the raw coordinates.
This is the convention Semester 2's gating mechanism will read directly, so
it must stay consistent from here on.

Raw dims (before the +1 presence flag):
  left_hand  : 21 landmarks x 3 (x, y, z)      = 63
  right_hand : 21 landmarks x 3 (x, y, z)      = 63
  pose       : 33 landmarks x 4 (x, y, z, vis) = 132
"""

import torch
import torch.nn as nn


class LandmarkStreamEncoder(nn.Module):
    """One MLP encoder for a single landmark stream (hand or pose)."""

    def __init__(self, raw_dim: int, out_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.raw_dim = raw_dim
        self.net = nn.Sequential(
            nn.Linear(raw_dim + 1, hidden_dim),  # +1 for the presence flag
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """
        coords:   (B, T, raw_dim)  -- zero-filled on frames where detection failed
        presence: (B, T, 1)        -- 1.0 if detected this frame, else 0.0
        """
        x = torch.cat([coords, presence], dim=-1)
        return self.net(x)  # (B, T, out_dim)


class LandmarkBranch(nn.Module):
    """Bundles the three landmark streams together for convenience."""

    def __init__(self, stream_dim: int = 64):
        super().__init__()
        self.left_hand = LandmarkStreamEncoder(raw_dim=63, out_dim=stream_dim)
        self.right_hand = LandmarkStreamEncoder(raw_dim=63, out_dim=stream_dim)
        self.pose = LandmarkStreamEncoder(raw_dim=132, out_dim=stream_dim)

    def forward(self, landmarks: dict) -> dict:
        """
        landmarks: {
          "left_hand":  {"coords": (B,T,63),  "presence": (B,T,1)},
          "right_hand": {"coords": (B,T,63),  "presence": (B,T,1)},
          "pose":       {"coords": (B,T,132), "presence": (B,T,1)},
        }
        Returns a dict of (B, T, stream_dim) tensors, kept SEPARATE
        (not concatenated here) so the fusion module -- and later,
        Semester 2's gating -- can act on each stream independently.
        """
        return {
            "left_hand": self.left_hand(
                landmarks["left_hand"]["coords"], landmarks["left_hand"]["presence"]
            ),
            "right_hand": self.right_hand(
                landmarks["right_hand"]["coords"], landmarks["right_hand"]["presence"]
            ),
            "pose": self.pose(
                landmarks["pose"]["coords"], landmarks["pose"]["presence"]
            ),
        }
