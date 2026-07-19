"""
Pairs preprocessing output (cached RGB .npy + landmark .npz) with gloss
labels for CTC training.

Expects, per split (train/dev/test):
    datasets/processed/<split>/<video_id>.npy               (T,224,224,3) uint8
    datasets/processed/<split>/<video_id>_landmarks.npz      from cache_landmarks.py
    datasets/processed/<split>/labels.json                   {video_id: [gloss_ids...]}
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SignLanguageDataset(Dataset):
    def __init__(self, processed_dir: str, split: str):
        self.split_dir = Path(processed_dir) / split

        with open(self.split_dir / "labels.json") as f:
            self.labels = json.load(f)

        self.video_ids = sorted(self.labels.keys())

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]

        rgb = np.load(self.split_dir / f"{video_id}.npy")            # (T,224,224,3) uint8
        rgb = torch.from_numpy(rgb).float().permute(0, 3, 1, 2) / 255.0  # (T,3,224,224)

        lm = np.load(self.split_dir / f"{video_id}_landmarks.npz")
        landmarks = {
            "left_hand": {
                "coords": torch.from_numpy(lm["left_hand_coords"]).float(),
                "presence": torch.from_numpy(lm["left_hand_presence"]).float(),
            },
            "right_hand": {
                "coords": torch.from_numpy(lm["right_hand_coords"]).float(),
                "presence": torch.from_numpy(lm["right_hand_presence"]).float(),
            },
            "pose": {
                "coords": torch.from_numpy(lm["pose_coords"]).float(),
                "presence": torch.from_numpy(lm["pose_presence"]).float(),
            },
        }

        target = torch.tensor(self.labels[video_id], dtype=torch.long)

        return rgb, landmarks, target


def collate_fn(batch):
    """
    Pads variable-length sequences to the batch max, and returns the
    unpadded lengths CTC needs (input_lengths, target_lengths).
    """
    rgbs, landmark_dicts, targets = zip(*batch)

    T_max = max(r.shape[0] for r in rgbs)
    input_lengths = torch.tensor([r.shape[0] for r in rgbs], dtype=torch.long)

    def pad_seq(t, T_max):
        pad_shape = (T_max - t.shape[0],) + t.shape[1:]
        return torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)

    rgb_batch = torch.stack([pad_seq(r, T_max) for r in rgbs])

    stream_names = ["left_hand", "right_hand", "pose"]
    landmark_batch = {}
    for stream in stream_names:
        coords = torch.stack([pad_seq(d[stream]["coords"], T_max) for d in landmark_dicts])
        presence = torch.stack([pad_seq(d[stream]["presence"], T_max) for d in landmark_dicts])
        landmark_batch[stream] = {"coords": coords, "presence": presence}

    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    target_concat = torch.cat(targets)  # CTC wants targets concatenated, not padded

    return rgb_batch, landmark_batch, target_concat, input_lengths, target_lengths
