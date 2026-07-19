"""
Run the full pipeline on a single raw video: extract frames + landmarks in
memory, run the model, print the predicted gloss sequence.

This is also the script to use for edge-deployment latency benchmarking
(the Sem 1 memory/latency table STNet's own paper never produced) --
wrap the model(...) call with timing, and run on your target device
(e.g. Jetson) with batch_size=1, as done here.

Usage:
    python inference.py --checkpoint checkpoints/epoch_60.pt --video path/to/clip.mp4
"""

import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

from models.model import SignMamba
from preprocessing.extract_frames import resize_and_crop
from preprocessing.cache_landmarks import STREAM_SPECS, flatten_stream

mp_holistic = mp.solutions.holistic


def process_video(video_path: str):
    """Runs RGB preprocessing + MediaPipe extraction + caching in one pass,
    mirroring the three preprocessing stages but in memory, for single-video
    inference rather than dataset-scale batch preprocessing."""
    cap = cv2.VideoCapture(video_path)
    frames, records = [], []

    with mp_holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            cropped = resize_and_crop(rgb, train=False)
            frames.append(cropped)

            records.append({
                "left_hand": [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                if results.left_hand_landmarks else None,
                "right_hand": [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                if results.right_hand_landmarks else None,
                "pose": [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                if results.pose_landmarks else None,
            })

    cap.release()

    rgb_arr = np.stack(frames).astype(np.float32) / 255.0
    rgb_tensor = torch.from_numpy(rgb_arr).permute(0, 3, 1, 2).unsqueeze(0)  # (1,T,3,224,224)

    landmarks = {}
    for stream_name, spec in STREAM_SPECS.items():
        coords, presence = [], []
        for record in records:
            flat, p = flatten_stream(record[stream_name], spec)
            coords.append(flat)
            presence.append([p])
        landmarks[stream_name] = {
            "coords": torch.tensor(np.stack(coords), dtype=torch.float32).unsqueeze(0),
            "presence": torch.tensor(np.stack(presence), dtype=torch.float32).unsqueeze(0),
        }

    return rgb_tensor, landmarks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SignMamba(
        vocab_size=cfg["data"]["vocab_size"],
        rgb_dim=cfg["model"]["rgb_dim"],
        stream_dim=cfg["model"]["stream_dim"],
        d_model=cfg["model"]["d_model"],
        mamba_depth_per_level=cfg["model"]["mamba_depth_per_level"],
        blank_id=cfg["data"]["blank_id"],
        pretrained_backbone=False,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rgb_tensor, landmarks = process_video(args.video)
    rgb_tensor = rgb_tensor.to(device)
    landmarks = {s: {k: v.to(device) for k, v in d.items()} for s, d in landmarks.items()}

    start = time.time()
    with torch.no_grad():
        log_probs = model(rgb_tensor, landmarks)
        pred_ids = model.decode(log_probs)[0]
    latency_ms = (time.time() - start) * 1000

    print(f"predicted gloss ids: {pred_ids}")
    print(f"inference latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
