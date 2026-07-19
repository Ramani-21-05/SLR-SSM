"""
Stage 3 of preprocessing: raw MediaPipe JSON -> fixed-shape cached tensors.

This is where the missing-landmark convention gets enforced, once, for the
whole project: any frame where MediaPipe failed to detect a stream gets a
zero vector for that stream's coordinates and a presence flag of 0.0.
Every downstream consumer (LandmarkStreamEncoder, and Semester 2's
confidence-aware fusion) relies on this convention being applied here and
nowhere else.

Output per video: a single .npz with, for each stream, a coords array
(T, raw_dim) and a presence array (T, 1) -- ready to load directly into
LandmarkBranch at train time, no further processing needed.

Usage:
    python preprocessing/cache_landmarks.py \
        --mediapipe_dir datasets/mediapipe \
        --out_dir datasets/processed \
        --split train
"""

import argparse
import json
from pathlib import Path

import numpy as np

STREAM_SPECS = {
    "left_hand": {"n_points": 21, "dims": 3},
    "right_hand": {"n_points": 21, "dims": 3},
    "pose": {"n_points": 33, "dims": 4},  # includes visibility
}


def flatten_stream(points, spec):
    """points: list of [x,y,z] or [x,y,z,vis] or None -> (flat_dim,), presence flag"""
    flat_dim = spec["n_points"] * spec["dims"]
    if points is None:
        return np.zeros(flat_dim, dtype=np.float32), 0.0
    arr = np.array(points, dtype=np.float32).reshape(-1)
    if arr.shape[0] != flat_dim:
        # Defensive: malformed detection, treat as missing rather than crash.
        return np.zeros(flat_dim, dtype=np.float32), 0.0
    return arr, 1.0


def cache_video(json_path: str, out_path: str):
    with open(json_path) as f:
        records = json.load(f)

    T = len(records)
    cached = {}

    for stream_name, spec in STREAM_SPECS.items():
        coords = np.zeros((T, spec["n_points"] * spec["dims"]), dtype=np.float32)
        presence = np.zeros((T, 1), dtype=np.float32)

        for t, record in enumerate(records):
            flat, p = flatten_stream(record.get(stream_name), spec)
            coords[t] = flat
            presence[t, 0] = p

        cached[f"{stream_name}_coords"] = coords
        cached[f"{stream_name}_presence"] = presence

    np.savez(out_path, **cached)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mediapipe_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = parser.parse_args()

    in_split_dir = Path(args.mediapipe_dir) / args.split
    out_split_dir = Path(args.out_dir) / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(in_split_dir.glob("*.json"))

    for json_path in json_files:
        out_path = out_split_dir / (json_path.stem + "_landmarks.npz")
        cache_video(str(json_path), str(out_path))
        print(f"cached: {json_path.name} -> {out_path.name}")


if __name__ == "__main__":
    main()
