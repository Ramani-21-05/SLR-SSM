"""
Training entry point.

Usage:
    python train.py --config configs/config.yaml
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.dataset import SignLanguageDataset, collate_fn
from models.model import SignMamba


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def move_landmarks_to_device(landmarks: dict, device):
    return {
        stream: {k: v.to(device) for k, v in parts.items()}
        for stream, parts in landmarks.items()
    }


def train_one_epoch(model, loader, optimizer, device, log_every: int):
    model.train()
    running_loss = 0.0

    for step, (rgb, landmarks, targets, input_lengths, target_lengths) in enumerate(loader):
        rgb = rgb.to(device)
        landmarks = move_landmarks_to_device(landmarks, device)
        targets = targets.to(device)

        optimizer.zero_grad()
        log_probs = model(rgb, landmarks)  # (T, B, vocab)

        loss = model.compute_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % log_every == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    return running_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = SignLanguageDataset(cfg["data"]["processed_dir"], split="train")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = SignMamba(
        vocab_size=cfg["data"]["vocab_size"],
        rgb_dim=cfg["model"]["rgb_dim"],
        stream_dim=cfg["model"]["stream_dim"],
        d_model=cfg["model"]["d_model"],
        mamba_depth_per_level=cfg["model"]["mamba_depth_per_level"],
        blank_id=cfg["data"]["blank_id"],
        pretrained_backbone=cfg["model"]["pretrained_backbone"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"]
    )

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, cfg["train"]["log_every"]
        )
        scheduler.step()
        print(f"epoch {epoch+1}/{cfg['train']['epochs']}  avg_loss {avg_loss:.4f}")

        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch+1}.pt")


if __name__ == "__main__":
    main()
