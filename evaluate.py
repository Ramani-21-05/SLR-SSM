"""
Evaluation entry point -- computes Word Error Rate (WER), same metric and
formula STNet uses (Eq. 8 in the paper), so results are directly comparable.

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/epoch_60.pt --split dev
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dataset import SignLanguageDataset, collate_fn
from models.model import SignMamba
from train import load_config, move_landmarks_to_device


def edit_distance(ref: list, hyp: list) -> int:
    """Standard Levenshtein distance, used to derive WER's S+D+I count."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]


def compute_wer(all_refs: list, all_hyps: list) -> float:
    total_errors = sum(edit_distance(r, h) for r, h in zip(all_refs, all_hyps))
    total_ref_len = sum(len(r) for r in all_refs)
    return 100.0 * total_errors / max(1, total_ref_len)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_refs, all_hyps = [], []

    offset = 0
    for rgb, landmarks, targets_concat, input_lengths, target_lengths in loader:
        rgb = rgb.to(device)
        landmarks = move_landmarks_to_device(landmarks, device)

        log_probs = model(rgb, landmarks)
        hyps = model.decode(log_probs)

        pos = 0
        for length in target_lengths.tolist():
            ref = targets_concat[pos: pos + length].tolist()
            all_refs.append(ref)
            pos += length
        all_hyps.extend(hyps)

    return compute_wer(all_refs, all_hyps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_set = SignLanguageDataset(cfg["data"]["processed_dir"], split=args.split)
    eval_loader = DataLoader(
        eval_set, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    model = SignMamba(
        vocab_size=cfg["data"]["vocab_size"],
        rgb_dim=cfg["model"]["rgb_dim"],
        stream_dim=cfg["model"]["stream_dim"],
        d_model=cfg["model"]["d_model"],
        mamba_depth_per_level=cfg["model"]["mamba_depth_per_level"],
        blank_id=cfg["data"]["blank_id"],
        pretrained_backbone=False,  # weights come from the checkpoint, not ImageNet
    ).to(device)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    wer = evaluate(model, eval_loader, device)
    print(f"{args.split} WER: {wer:.2f}%")


if __name__ == "__main__":
    main()
