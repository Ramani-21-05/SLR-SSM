"""
Stage 2 of preprocessing: raw video -> per-frame MediaPipe landmarks.

Runs MediaPipe Holistic (hands + pose; face skipped in Sem 1, see
architecture notes) over each video and writes ONE raw landmark record per
frame. This is intentionally separate from cache_landmarks.py: this script
does detection only, the next stage turns detections into fixed-shape
tensors with presence flags for training.

Usage:
    python preprocessing/mediapipe_extract.py \
        --raw_dir datasets/raw \
        --out_dir datasets/mediapipe \
        --split train
"""

import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic


def landmarks_to_list(landmark_list, with_visibility: bool = False):
    if landmark_list is None:
        return None
    pts = []
    for lm in landmark_list.landmark:
        if with_visibility:
            pts.append([lm.x, lm.y, lm.z, lm.visibility])
        else:
            pts.append([lm.x, lm.y, lm.z])
    return pts


def extract_video(video_path: str, out_path: str, holistic):
    cap = cv2.VideoCapture(video_path)
    per_frame_records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        record = {
            "left_hand": landmarks_to_list(results.left_hand_landmarks),
            "right_hand": landmarks_to_list(results.right_hand_landmarks),
            "pose": landmarks_to_list(results.pose_landmarks, with_visibility=True),
        }
        per_frame_records.append(record)

    cap.release()

    with open(out_path, "w") as f:
        json.dump(per_frame_records, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = parser.parse_args()

    raw_split_dir = Path(args.raw_dir) / args.split
    out_split_dir = Path(args.out_dir) / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(raw_split_dir.glob("*.mp4"))

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for video_path in video_files:
            out_path = out_split_dir / (video_path.stem + ".json")
            extract_video(str(video_path), str(out_path), holistic)
            print(f"extracted landmarks: {video_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
