#!/usr/bin/env python3
"""
L&T CCTV AI - Stage 2: Offline Action Recognition Test

Tests a trained action recognition model on a video file.

Usage:
    python test_action.py --video path/to/video.mp4
    python test_action.py --video path/to/video.mp4 --output output.mp4
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure onnxruntime can find CUDA DLLs
def _setup_cuda_path():
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib) and torch_lib not in os.environ.get('PATH', ''):
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
    except ImportError:
        pass
_setup_cuda_path()
del _setup_cuda_path

import cv2
import numpy as np
import torch

from src.action.config import (
    ACTION_CLASSES,
    MODEL_SAVE_DIR,
    NUM_FRAMES,
    get_status,
)
from src.action.inference import ActionRecognizer, TemporalActionState

# ── Logging Setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device() -> str:
    """Determine inference device."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: CUDA — GPU: {gpu_name}")
        return "cuda"
    else:
        logger.warning("Device: CPU — No CUDA GPU available.")
        return "cpu"


def test_video(
    video_path: str,
    output_path: str = None,
    checkpoint_path: Path = None,
    device: str = None,
):
    """
    Test action recognition on a video file.

    Args:
        video_path: Path to the input video.
        output_path: Optional path for annotated output video.
        checkpoint_path: Path to model checkpoint. If None, uses best.
        device: Inference device.
    """
    if not os.path.isfile(video_path):
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)

    if device is None:
        device = get_device()

    print()
    print("=" * 60)
    print("  L&T CCTV AI - Stage 2: Offline Action Test")
    print("=" * 60)
    print()
    print(f"  Video:     {video_path}")
    print(f"  Device:    {device.upper()}")
    print()

    # Load model
    recognizer = ActionRecognizer(
        device=device,
        checkpoint_path=checkpoint_path,
    )
    recognizer.load()

    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps:.1f}")
    print(f"  Frames:     {total_frames}")
    print(f"  Duration:   {duration:.1f}s")
    print()

    # ── Single-window prediction (full video) ──────────────────────────────
    print("Running full-video prediction...")
    start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

    if device == "cuda" and start_time:
        start_time.record()
    t_start = __import__("time").perf_counter()

    prediction = recognizer.predict_with_temporal_smoothing(
        video_path,
        num_frames=NUM_FRAMES,
        frame_stride=2,
        window_stride=8,
    )

    if device == "cuda" and end_time:
        end_time.record()
        torch.cuda.synchronize()
        inference_time_ms = start_time.elapsed_time(end_time)
    else:
        inference_time_ms = (__import__("time").perf_counter() - t_start) * 1000

    print()
    print("─" * 60)
    print(f"  ACTION:     {prediction.action}")
    print(f"  CONFIDENCE: {prediction.confidence * 100:.1f}%")
    print(f"  STATUS:     {prediction.status}")
    print(f"  TIME:       {inference_time_ms:.1f}ms")
    print("─" * 60)
    print()

    # Print per-class probabilities
    if prediction.all_probs is not None:
        print("  Per-class probabilities:")
        sorted_indices = np.argsort(-prediction.all_probs)
        for idx in sorted_indices[:5]:  # Top 5
            name = ACTION_CLASSES[idx]
            prob = prediction.all_probs[idx]
            bar = "█" * int(prob * 30)
            print(f"    {name:>12s}: {prob * 100:5.1f}% {bar}")
        print()

    # ── Annotated output video (optional) ──────────────────────────────────
    if output_path:
        print(f"Creating annotated output video: {output_path}")
        _create_annotated_video(
            video_path, output_path, recognizer, fps, device
        )
        print(f"  Saved to: {output_path}")
        print()

    print("Test complete.")


def _create_annotated_video(
    input_path: str,
    output_path: str,
    recognizer: ActionRecognizer,
    fps: float,
    device: str,
):
    """Create an annotated output video with action predictions overlaid."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Rolling buffer for temporal smoothing
    buffer_size = NUM_FRAMES
    frame_buffer = []
    action_state = TemporalActionState()
    frame_idx = 0
    action_interval = 5  # Run action model every N frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add to buffer
        frame_buffer.append(frame_rgb)
        if len(frame_buffer) > buffer_size:
            frame_buffer.pop(0)

        # Run action model periodically
        if frame_idx % action_interval == 0 and len(frame_buffer) >= buffer_size:
            frames_arr = np.stack(frame_buffer, axis=0)
            pred = recognizer.predict_video_tensor(frames_arr)
            action_state.update(pred.action, pred.confidence)

        # Get current smoothed prediction
        current_action, current_conf = action_state._stable_action, action_state._stable_confidence
        current_status = get_status(current_action)

        # Draw overlay
        overlay = frame.copy()

        # Action info box
        box_h = 100
        box_w = 350
        box_x = width - box_w - 10
        box_y = 10

        # Background
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 1)

        # Action text
        status_color = (0, 0, 255) if current_status == "SUSPICIOUS" else (0, 200, 0)
        cv2.putText(overlay, f"Action: {current_action}", (box_x + 10, box_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Conf: {current_conf * 100:.1f}%", (box_x + 10, box_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Status: {current_status}", (box_x + 10, box_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        # Frame counter
        cv2.putText(overlay, f"Frame: {frame_idx}/{total_frames}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(overlay)

    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(
        description="L&T CCTV AI - Stage 2: Offline Action Recognition Test"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for annotated output video (optional).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. Default: models/action_model/best_model.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Inference device. Default: auto-detect.",
    )

    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None

    test_video(
        video_path=args.video,
        output_path=args.output,
        checkpoint_path=checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
