"""
Action Inference Module (Stage 2)

Reusable action recognition inference for both webcam and offline use.
Handles model loading, frame preprocessing, prediction, and temporal smoothing.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.action.config import (
    ACTION_CLASSES,
    ACTION_SMOOTHING_MIN,
    ACTION_SMOOTHING_WINDOW,
    MODEL_SAVE_DIR,
    NUM_ACTION_CLASSES,
    NUM_FRAMES,
    get_status,
)
from src.action.dataset import normalize_frames, sample_frames_deterministic
from src.action.model import load_trained_model

logger = logging.getLogger(__name__)


@dataclass
class ActionPrediction:
    """Result of an action recognition prediction."""

    action: str           # Display name of the predicted action (e.g., "Walk")
    confidence: float     # Prediction confidence (0.0 to 1.0)
    status: str           # "NORMAL" or "SUSPICIOUS"
    action_idx: int       # Index of the predicted action
    all_probs: Optional[np.ndarray] = None  # Full probability distribution


@dataclass
class TemporalActionState:
    """
    Maintains temporal smoothing for action predictions.
    Uses a sliding window of recent predictions to prevent flickering.
    """

    window_size: int = ACTION_SMOOTHING_WINDOW
    min_count: int = ACTION_SMOOTHING_MIN

    _history: Deque[Tuple[str, float]] = None
    _stable_action: str = "Stand"
    _stable_confidence: float = 0.0

    def __post_init__(self):
        if self._history is None:
            self._history = deque(maxlen=self.window_size)

    def update(self, action: str, confidence: float) -> Tuple[str, float]:
        """
        Add a new prediction and return the smoothed result.

        Args:
            action: Current predicted action name.
            confidence: Current prediction confidence.

        Returns:
            (smoothed_action, smoothed_confidence)
        """
        self._history.append((action, confidence))

        # Count occurrences in history
        action_counts: Dict[str, int] = {}
        action_confs: Dict[str, List[float]] = {}
        for act, conf in self._history:
            action_counts[act] = action_counts.get(act, 0) + 1
            action_confs.setdefault(act, []).append(conf)

        # Find most frequent action
        most_frequent = max(action_counts, key=lambda k: action_counts[k])
        count = action_counts[most_frequent]
        avg_conf = sum(action_confs[most_frequent]) / len(action_confs[most_frequent])

        # Only switch if we have enough consensus or not enough history yet
        if count >= self.min_count or len(self._history) < self.window_size:
            self._stable_action = most_frequent
            self._stable_confidence = avg_conf

        return self._stable_action, self._stable_confidence

    def reset(self):
        """Reset the smoothing state."""
        self._history.clear()
        self._stable_action = "Stand"
        self._stable_confidence = 0.0


class ActionRecognizer:
    """
    Action recognition inference engine.

    Loads a trained model and provides prediction methods for both
    individual videos and real-time webcam streams.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        """
        Args:
            device: Inference device ("cuda", "cpu", or None for auto).
            checkpoint_path: Path to trained model checkpoint. If None, uses best.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.checkpoint_path = checkpoint_path
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.config = None
        self._loaded = False

    def load(self) -> str:
        """
        Load the trained model.

        Returns:
            Device string where model is loaded.
        """
        if self._loaded:
            return self.device

        logger.info(f"Loading action recognition model on {self.device}...")

        self.model, self.class_to_idx, self.config = load_trained_model(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        # Build reverse mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self._loaded = True
        logger.info(
            f"Action model loaded: {len(self.class_to_idx)} classes on {self.device}"
        )
        return self.device

    def _preprocess_video_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frames into model input tensor.

        VideoMAE expects (batch, num_frames, num_channels, H, W).

        Args:
            frames: (T, H, W, 3) uint8 RGB frames.

        Returns:
            Tensor of shape (1, T, 3, H, W) ready for model input.
        """
        frame_size = self.config.get("frame_size", 224) if self.config else 224
        T, H, W, C = frames.shape

        if H != frame_size or W != frame_size:
            resized = []
            for i in range(T):
                resized.append(
                    cv2.resize(frames[i], (frame_size, frame_size))
                )
            frames = np.stack(resized, axis=0)

        # Normalize — returns (T, 3, H, W)
        tensor = normalize_frames(frames)

        # Add batch dimension -> (1, T, 3, H, W)
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    @torch.no_grad()
    def predict_video_tensor(self, frames: np.ndarray) -> ActionPrediction:
        """
        Predict action from pre-loaded video frames.

        Args:
            frames: (T, H, W, 3) uint8 RGB frames.

        Returns:
            ActionPrediction with predicted action, confidence, and status.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        tensor = self._preprocess_video_tensor(frames)

        # Forward pass
        outputs = self.model(pixel_values=tensor)
        logits = outputs.logits  # (1, num_classes)

        # Softmax probabilities
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # Get prediction
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        action_name = self.idx_to_class[pred_idx]
        status = get_status(action_name)

        return ActionPrediction(
            action=action_name,
            confidence=confidence,
            status=status,
            action_idx=pred_idx,
            all_probs=probs,
        )

    def predict_video_file(
        self,
        video_path: str,
        num_frames: int = NUM_FRAMES,
        frame_stride: int = 2,
    ) -> ActionPrediction:
        """
        Predict action from a video file.

        Args:
            video_path: Path to the video file.
            num_frames: Number of frames to sample.
            frame_stride: Temporal stride.

        Returns:
            ActionPrediction.
        """
        frames = sample_frames_deterministic(
            video_path,
            num_frames=num_frames,
            frame_stride=frame_stride,
        )

        if frames is None:
            logger.error(f"Could not read video: {video_path}")
            return ActionPrediction(
                action="Unknown",
                confidence=0.0,
                status="NORMAL",
                action_idx=-1,
            )

        return self.predict_video_tensor(frames)

    def predict_with_temporal_smoothing(
        self,
        video_path: str,
        num_frames: int = NUM_FRAMES,
        frame_stride: int = 2,
        window_stride: int = 8,
    ) -> ActionPrediction:
        """
        Predict action from a video using overlapping temporal windows
        with smoothing for more stable predictions.

        Args:
            video_path: Path to the video.
            num_frames: Frames per window.
            frame_stride: Temporal stride within a window.
            window_stride: Stride between windows (in frames).

        Returns:
            ActionPrediction with temporally smoothed prediction.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ActionPrediction(
                action="Unknown",
                confidence=0.0,
                status="NORMAL",
                action_idx=-1,
            )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames < num_frames:
            # Video too short, predict once
            return self.predict_video_file(video_path, num_frames, frame_stride)

        # Sample multiple windows
        all_predictions = []
        window_indices = range(
            0,
            max(1, total_frames - num_frames * frame_stride),
            window_stride * frame_stride,
        )

        for start_frame in window_indices:
            cap = cv2.VideoCapture(video_path)
            frame_indices = [
                min(start_frame + i * frame_stride, total_frames - 1)
                for i in range(num_frames)
            ]

            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            if len(frames) >= num_frames:
                frames_arr = np.stack(frames[:num_frames], axis=0)
                pred = self.predict_video_tensor(frames_arr)
                all_predictions.append(pred)

        if not all_predictions:
            return ActionPrediction(
                action="Unknown",
                confidence=0.0,
                status="NORMAL",
                action_idx=-1,
            )

        # Average probabilities across windows
        avg_probs = np.mean(
            np.stack([p.all_probs for p in all_predictions]), axis=0
        )

        pred_idx = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])
        action_name = self.idx_to_class[pred_idx]
        status = get_status(action_name)

        return ActionPrediction(
            action=action_name,
            confidence=confidence,
            status=status,
            action_idx=pred_idx,
            all_probs=avg_probs,
        )
