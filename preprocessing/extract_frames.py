"""
Stage 1 of preprocessing: raw video -> resized/cropped RGB frame tensors.

Follows the CorrNet/STNet convention so results stay comparable to their
published numbers: resize to 256x256, then either random-crop to 224x224
(training, with horizontal flip) or center-crop to 224x224 (inference).

Usage:
    python preprocessing/extract_frames.py \
        --raw_dir datasets/raw \
        --out_dir datasets/processed \
        --split train
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def resize_and_crop(frame: np.ndarray, train: bool, crop_size: int = 224) -> np.ndarray:
    frame = cv2.resize(frame, (256, 256))

    if train:
        top = np.random.randint(0, 256 - crop_size + 1)
        left = np.random.randint(0, 256 - crop_size + 1)
        if np.random.rand() < 0.5:
            frame = cv2.flip(frame, 1)
    else:
        top = left = (256 - crop_size) // 2

    return frame[top: top + crop_size, left: left + crop_size]


def extract_video(video_path: str, out_path: str, train: bool):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_and_crop(frame, train=train)
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"[warn] no frames read from {video_path}")
        return

    arr = np.stack(frames).astype(np.uint8)  # (T, 224, 224, 3)
    np.save(out_path, arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = parser.parse_args()

    raw_split_dir = Path(args.raw_dir) / args.split
    out_split_dir = Path(args.out_dir) / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    is_train = args.split == "train"
    video_files = sorted(raw_split_dir.glob("*.mp4"))

    for video_path in video_files:
        out_path = out_split_dir / (video_path.stem + ".npy")
        extract_video(str(video_path), str(out_path), train=is_train)
        print(f"processed {video_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
