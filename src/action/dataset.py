"""
Video Action Dataset Loader (Stage 2)

Loads videos from the dataset, samples frames, applies transforms,
and provides PyTorch Dataset interface for training and inference.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.action.config import (
    ACTION_CLASS_TO_DIR,
    ACTION_CLASSES,
    DATASET_DIR,
    FRAME_SIZE,
    NUM_FRAMES,
    NUM_ACTION_CLASSES,
    SPLITS_DIR,
    SPLIT_FOLD,
    SPLIT_PERCENTAGE,
)

logger = logging.getLogger(__name__)


def _load_split_files(
    action_dir_name: str,
    split_percentage: str,
    split_fold: int,
) -> set:
    """
    Load test video filenames from the provided split files.

    Args:
        action_dir_name: Directory name of the action (e.g., "fall").
        split_percentage: "50%" or "75%".
        split_fold: Fold number (1-5).

    Returns:
        Set of video filenames that are in the TEST split.
    """
    split_file = (
        SPLITS_DIR
        / split_percentage
        / f"{action_dir_name}_test_split{split_fold}.txt"
    )

    test_videos = set()
    if not split_file.exists():
        logger.warning(f"Split file not found: {split_file}")
        return test_videos

    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 1:
                video_name = parts[0].strip()
                test_videos.add(video_name)

    return test_videos


def discover_dataset(
    dataset_dir: Optional[Path] = None,
    split_percentage: str = SPLIT_PERCENTAGE,
    split_fold: int = SPLIT_FOLD,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Discover all videos and split them into train/val using provided split files.

    Returns:
        (train_samples, val_samples) where each sample is a dict with keys:
        'path', 'label' (display name), 'label_idx', 'video_name'
    """
    if dataset_dir is None:
        dataset_dir = DATASET_DIR

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    # Build class-to-index mapping
    class_to_idx = {name: idx for idx, name in enumerate(ACTION_CLASSES)}

    train_samples = []
    val_samples = []
    class_counts = {name: {"train": 0, "val": 0} for name in ACTION_CLASSES}

    # Track all found classes
    found_classes = set()

    for action_name, dir_name in ACTION_CLASS_TO_DIR.items():
        action_path = dataset_dir / dir_name
        if not action_path.exists():
            logger.warning(f"Action directory not found: {action_path}")
            continue

        found_classes.add(action_name)

        # Load test split for this action
        test_videos = _load_split_files(dir_name, split_percentage, split_fold)

        # Find all video files
        video_files = []
        for f in sorted(action_path.iterdir()):
            if f.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                video_files.append(f)

        if not video_files:
            logger.warning(f"No video files found in {action_path}")
            continue

        for video_path in video_files:
            sample = {
                "path": str(video_path),
                "label": action_name,
                "label_idx": class_to_idx[action_name],
                "video_name": video_path.name,
            }

            if video_path.name in test_videos:
                val_samples.append(sample)
                class_counts[action_name]["val"] += 1
            else:
                train_samples.append(sample)
                class_counts[action_name]["train"] += 1

    # Report dataset stats
    logger.info(f"Dataset discovered: {len(train_samples)} train, {len(val_samples)} val")
    logger.info("Per-class distribution:")
    for name in ACTION_CLASSES:
        counts = class_counts.get(name, {"train": 0, "val": 0})
        logger.info(f"  {name:12s}: train={counts['train']:4d}, val={counts['val']:4d}")

    # Check for missing classes
    missing = set(ACTION_CLASSES) - found_classes
    if missing:
        logger.warning(f"Missing action classes in dataset: {missing}")

    return train_samples, val_samples


def sample_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    frame_stride: int = 2,
) -> Optional[np.ndarray]:
    """
    Sample frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.
        frame_stride: Temporal stride between sampled frames.

    Returns:
        numpy array of shape (T, H, W, 3) in RGB, or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logger.warning(f"Video has 0 frames: {video_path}")
        cap.release()
        return None

    # Calculate frame indices to sample
    # Sample every `frame_stride` frames, then take `num_frames` evenly spaced
    max_start = max(0, total_frames - num_frames * frame_stride)
    start_idx = random.randint(0, max_start) if max_start > 0 else 0

    frame_indices = [
        min(start_idx + i * frame_stride, total_frames - 1)
        for i in range(num_frames)
    ]

    # Read frames
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if len(frames) < num_frames:
        # Pad with last frame if we didn't get enough
        if frames:
            while len(frames) < num_frames:
                frames.append(frames[-1].copy())
        else:
            logger.warning(f"Could not read any frames from: {video_path}")
            return None

    return np.stack(frames, axis=0)  # (T, H, W, 3)


def sample_frames_deterministic(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    frame_stride: int = 2,
) -> Optional[np.ndarray]:
    """
    Deterministically sample frames from a video (for validation/inference).
    Always starts from the beginning.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # Always start from frame 0 for determinism
    frame_indices = [
        min(i * frame_stride, total_frames - 1)
        for i in range(num_frames)
    ]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    if len(frames) < num_frames:
        if frames:
            while len(frames) < num_frames:
                frames.append(frames[-1].copy())
        else:
            return None

    return np.stack(frames, axis=0)


def apply_train_transform(frames: np.ndarray) -> np.ndarray:
    """
    Apply training augmentations to a batch of frames.

    Args:
        frames: (T, H, W, 3) uint8 RGB frames.

    Returns:
        Transformed frames (T, H, W, 3) uint8.
    """
    T, H, W, C = frames.shape

    # Random horizontal flip (does not invalidate most actions)
    if random.random() > 0.5:
        frames = frames[:, :, ::-1, :].copy()

    # Random brightness/contrast jitter
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        frames = np.clip(frames * factor, 0, 255).astype(np.uint8)

    # Random spatial crop (take center 90% then resize back)
    if random.random() > 0.5:
        crop_h = int(H * 0.9)
        crop_w = int(W * 0.9)
        y_start = random.randint(0, H - crop_h)
        x_start = random.randint(0, W - crop_w)
        frames = frames[:, y_start:y_start + crop_h, x_start:x_start + crop_w, :]
        # Resize back
        resized = []
        for i in range(T):
            resized.append(cv2.resize(frames[i], (W, H)))
        frames = np.stack(resized, axis=0)

    return frames


def normalize_frames(
    frames: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Normalize frames to [0, 1] float and apply ImageNet normalization.

    VideoMAE expects input of shape (batch, num_frames, num_channels, H, W),
    so we output (T, 3, H, W) so that when batched we get (B, T, 3, H, W).

    Args:
        frames: (T, H, W, 3) uint8 RGB frames.
        mean: Per-channel mean for normalization.
        std: Per-channel std for normalization.

    Returns:
        Tensor of shape (T, 3, H, W) normalized float32.
    """
    # Convert to float [0, 1]
    tensor = torch.from_numpy(frames).float() / 255.0

    # Rearrange to (T, H, W, 3) -> (T, 3, H, W)
    tensor = tensor.permute(0, 3, 1, 2)

    # Normalize with ImageNet stats
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    tensor = (tensor - mean_t) / std_t

    return tensor


class ActionVideoDataset(Dataset):
    """
    PyTorch Dataset for action recognition videos.

    Loads videos, samples frames, applies transforms, and returns
    (video_tensor, label_index) pairs.
    """

    def __init__(
        self,
        samples: List[Dict],
        num_frames: int = NUM_FRAMES,
        frame_stride: int = 2,
        frame_size: int = FRAME_SIZE,
        is_training: bool = True,
    ):
        """
        Args:
            samples: List of sample dicts from discover_dataset().
            num_frames: Number of frames to sample per video.
            frame_stride: Temporal stride for sampling.
            frame_size: Resize all frames to this size (square).
            is_training: If True, apply random augmentations.
        """
        self.samples = samples
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.frame_size = frame_size
        self.is_training = is_training

        # Pre-filter valid samples
        self.valid_samples = []
        for s in samples:
            if os.path.isfile(s["path"]):
                self.valid_samples.append(s)
            else:
                logger.warning(f"Video not found, skipping: {s['path']}")

        logger.info(
            f"Dataset initialized: {len(self.valid_samples)} valid / "
            f"{len(samples)} total samples"
        )

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.valid_samples[idx]
        video_path = sample["path"]
        label_idx = sample["label_idx"]

        # Sample frames
        if self.is_training:
            frames = sample_frames(
                video_path,
                num_frames=self.num_frames,
                frame_stride=self.frame_stride,
            )
        else:
            frames = sample_frames_deterministic(
                video_path,
                num_frames=self.num_frames,
                frame_stride=self.frame_stride,
            )

        # Fallback: create blank frames if video couldn't be read
        if frames is None:
            logger.warning(f"Using blank frames for: {video_path}")
            frames = np.zeros(
                (self.num_frames, self.frame_size, self.frame_size, 3),
                dtype=np.uint8,
            )

        T, H, W, C = frames.shape

        # Resize spatially
        resized_frames = []
        for i in range(T):
            resized = cv2.resize(
                frames[i], (self.frame_size, self.frame_size)
            )
            resized_frames.append(resized)
        frames = np.stack(resized_frames, axis=0)

        # Apply training augmentations
        if self.is_training:
            frames = apply_train_transform(frames)

        # Normalize to tensor
        tensor = normalize_frames(frames)  # (3, T, H, W)

        return tensor, label_idx


def collate_fn(batch):
    """Custom collate that stacks video tensors and labels."""
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), torch.tensor(labels, dtype=torch.long)
