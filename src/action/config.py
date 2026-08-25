"""
Action Recognition Configuration (Stage 2)

Centralized configuration for action classes, suspicious/normal mapping,
model settings, and training parameters.
"""

from pathlib import Path
from typing import Dict, List, Set

# ── Project Root ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Action Classes ─────────────────────────────────────────────────────────
# The 13 action classes from the dataset.
# Directory names in archive/Videos/Videos/ use underscores (e.g., "lying_down").
ACTION_CLASSES = [
    "Fall",
    "Grab",
    "Gun",
    "Hit",
    "Kick",
    "LyingDown",
    "Run",
    "Sit",
    "Stand",
    "Sneak",
    "Struggle",
    "Throw",
    "Walk",
]

# Map display name -> directory name in the dataset
ACTION_CLASS_TO_DIR: Dict[str, str] = {
    "Fall": "fall",
    "Grab": "grab",
    "Gun": "gun",
    "Hit": "hit",
    "Kick": "kick",
    "LyingDown": "lying_down",
    "Run": "run",
    "Sit": "sit",
    "Stand": "stand",
    "Sneak": "sneak",
    "Struggle": "struggle",
    "Throw": "throw",
    "Walk": "walk",
}

# Reverse map: directory name -> display name
ACTION_DIR_TO_CLASS: Dict[str, str] = {v: k for k, v in ACTION_CLASS_TO_DIR.items()}

# Number of action classes
NUM_ACTION_CLASSES = len(ACTION_CLASSES)

# ── Suspicious / Normal Mapping ───────────────────────────────────────────
# Actions classified as suspicious
SUSPICIOUS_ACTIONS: Set[str] = {
    "Fall",
    "Grab",
    "Gun",
    "Hit",
    "Kick",
    "LyingDown",
    "Run",
    "Struggle",
    "Throw",
}

# Actions classified as normal
NORMAL_ACTIONS: Set[str] = {
    "Walk",
    "Sit",
    "Stand",
    "Sneak",
}


def is_suspicious(action_name: str) -> bool:
    """Check if an action is classified as suspicious."""
    return action_name in SUSPICIOUS_ACTIONS


def get_status(action_name: str) -> str:
    """Get NORMAL or SUSPICIOUS status for an action."""
    return "SUSPICIOUS" if action_name in SUSPICIOUS_ACTIONS else "NORMAL"


# ── Dataset Paths ─────────────────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "archive" / "Videos" / "Videos"
SPLITS_DIR = PROJECT_ROOT / "archive" / "Test_Train_Splits"

# Use the 50% train / 50% test split (split1 as validation)
# The directory names refer to TEST percentage, not train
# "50%" = 50% test / 50% train
# "75%" = 75% test / 25% train
SPLIT_PERCENTAGE = "50%"
SPLIT_FOLD = 1  # Which fold to use for train/val split

# ── Model Configuration ───────────────────────────────────────────────────
# VideoMAE model from HuggingFace
MODEL_NAME = "MCG-NJU/videomae-base"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "action_model"
RESULTS_DIR = PROJECT_ROOT / "results"

# Video sampling
NUM_FRAMES = 16          # Number of frames to sample per video clip
FRAME_SIZE = 224         # Spatial resolution (VideoMAE expects 224x224)
FRAME_STRIDE = 2         # Temporal stride for frame sampling

import os

# ── Training Configuration ────────────────────────────────────────────────
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
NUM_EPOCHS = 30
WARMUP_EPOCHS = 5
NUM_WORKERS = 0 if os.name == "nt" else 2

# Early stopping
PATIENCE = 7  # Stop if no improvement for N epochs

# Checkpoint frequency
CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs

# ── Inference Configuration ───────────────────────────────────────────────
# Confidence threshold below which action is shown as low-confidence
ACTION_CONFIDENCE_THRESHOLD = 0.3

# Temporal smoothing for inference
ACTION_SMOOTHING_WINDOW = 5  # Number of predictions to smooth over
ACTION_SMOOTHING_MIN = 3     # Minimum count before switching prediction

# Webcam action inference interval (run action model every N frames)
WEBCAM_ACTION_INTERVAL = 5  # Run action model every 5 frames for real-time performance

# Rolling buffer size for webcam action recognition
WEBCAM_BUFFER_SIZE = NUM_FRAMES  # Keep NUM_FRAMES in the buffer
