# SignMamba -- Semester 1

RGB + MediaPipe fusion, lightweight backbone, hierarchical multi-scale
Mamba decoder, CTC loss. Built as a direct answer to STNet's own stated
limitations (GPU memory / batch-size-2 ceiling, and setting up the
one-hand-bias fix that Semester 2 will complete).

## Project structure

```
SignMamba/
├── datasets/
│   ├── raw/            # original .mp4 video files, one folder per split
│   ├── mediapipe/       # raw per-frame landmark JSON, from mediapipe_extract.py
│   ├── processed/       # final cached tensors ready for training
│   └── dataset.py       # PyTorch Dataset + collate_fn pairing frames+landmarks+labels
│
├── models/
│   ├── mobilenet.py         # RGB branch: MobileNetV3-Small encoder
│   ├── landmark_encoder.py  # MediaPipe branch: 3 separate stream MLPs (LH/RH/pose)
│   ├── fusion.py             # Sem 1 simple fusion: concat + linear
│   ├── mamba.py               # core selective-SSM block (pure PyTorch)
│   ├── pyramid_mamba.py       # hierarchical multi-scale decoder built from mamba.py
│   ├── ctc_head.py            # classifier + CTC loss + greedy decode
│   └── model.py               # assembles all of the above into SignMamba
│
├── preprocessing/
│   ├── extract_frames.py     # stage 1: raw video -> resized/cropped RGB .npy
│   ├── mediapipe_extract.py  # stage 2: raw video -> raw landmark .json
│   └── cache_landmarks.py    # stage 3: raw landmark json -> fixed-shape .npz
│                                (this is where missing-landmark handling lives)
│
├── configs/config.yaml   # all hyperparameters and paths in one place
├── train.py               # training loop
├── evaluate.py             # WER computation (same formula as STNet's Eq. 8)
└── inference.py             # single-video end-to-end run + latency timing
```

## What each stage does

**Preprocessing (run once per dataset):**
1. `extract_frames.py` -- reads raw `.mp4`s, resizes to 256x256, crops to
   224x224 (random crop + flip for train, center crop for dev/test), saves
   as `.npy`.
2. `mediapipe_extract.py` -- runs MediaPipe Holistic over each video,
   dumps raw per-frame hand/pose landmark detections to `.json`. Detection
   only -- no shape normalization yet.
3. `cache_landmarks.py` -- converts the raw JSON into fixed-shape arrays
   per stream (left hand, right hand, pose), zero-filling any frame where
   detection failed and recording a presence flag (1.0/0.0) for that frame.
   This presence-flag convention is what Semester 2's confidence-aware
   fusion will read directly, so it's centralized here rather than handled
   ad hoc downstream.

Run all three per split (`train`, `dev`, `test`) before training.

**Model forward pass (`models/model.py`):**
1. RGB frames go through `RGBEncoder` (MobileNetV3-Small) -> one 256-dim
   vector per frame.
2. Each landmark stream goes through its own small MLP in
   `LandmarkBranch` -> three separate 64-dim vectors per frame (kept
   separate, not merged, for Sem 2's sake).
3. `SimpleFusion` concatenates all four vectors and projects to
   `d_model` (512) with one linear layer -- deliberately simple in Sem 1.
4. `HierarchicalMultiScaleMamba` processes the fused sequence at three
   temporal resolutions (full, half, quarter) with its own Mamba stack at
   each level, then merges them back together with a residual connection.
5. `CTCHead` projects to vocab size, produces log-probs, and computes CTC
   loss during training or greedy-decodes to a gloss sequence at inference.

**Training (`train.py`):** loads the cached dataset, runs the forward
pass + CTC loss + backprop loop, checkpoints every epoch.

**Evaluation (`evaluate.py`):** loads a checkpoint, runs greedy decoding
on a split, computes WER using the same S+D+I / N_ref formula as STNet.

**Inference (`inference.py`):** takes one raw video, runs all three
preprocessing stages in memory (no disk caching), predicts a gloss
sequence, and times the forward pass -- use this for the edge-deployment
latency/memory table.

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
# 1. Preprocess each split
python preprocessing/extract_frames.py --raw_dir datasets/raw --out_dir datasets/processed --split train
python preprocessing/mediapipe_extract.py --raw_dir datasets/raw --out_dir datasets/mediapipe --split train
python preprocessing/cache_landmarks.py --mediapipe_dir datasets/mediapipe --out_dir datasets/processed --split train
# (repeat for dev, test)

# 2. Add datasets/processed/<split>/labels.json mapping video_id -> [gloss_ids...]

# 3. Train
python train.py --config configs/config.yaml

# 4. Evaluate
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/epoch_60.pt --split dev

# 5. Single-video inference / latency benchmark
python inference.py --checkpoint checkpoints/epoch_60.pt --video path/to/clip.mp4
```

## Known gaps / not yet done

- `models/mamba.py` is a from-scratch pure-PyTorch selective-scan block,
  not the official `mamba-ssm` CUDA kernel -- correct, but not optimized
  for training throughput. Swappable later without touching
  `pyramid_mamba.py`'s interface.
- No beam search decoder yet (`evaluate.py` uses greedy decoding) --
  add one before reporting final WER numbers for the paper.
- `labels.json` / vocabulary building is not included -- depends on which
  ASL dataset you finalize (How2Sign / OpenASL) and its gloss vocabulary.
- No data augmentation beyond crop+flip -- STNet also uses 20% temporal
  re-scaling during training, worth adding for a fair comparison.
