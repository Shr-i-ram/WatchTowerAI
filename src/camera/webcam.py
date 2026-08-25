"""
Webcam Capture Module

Manages OpenCV video capture with configurable resolution and FPS.
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class WebcamCapture:
    """
    Manages webcam video capture using OpenCV.

    Provides a clean interface for grabbing frames with resolution
    and FPS control.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
    ):
        """
        Initialize webcam capture.

        Args:
            camera_index: Camera device index (0 for default webcam).
            width: Desired capture width.
            height: Desired capture height.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """
        Open the webcam.

        Returns:
            True if webcam opened successfully.
        """
        logger.info(f"Opening webcam (index={self.camera_index})...")
        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(
                f"Failed to open webcam at index {self.camera_index}. "
                "Check that a camera is connected and not in use."
            )
            return False

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Read actual resolution after setting
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Webcam opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the webcam.

        Returns:
            Tuple of (success, frame). Frame is BGR numpy array or None.
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from webcam.")
            return False, None

        return True, frame

    def close(self) -> None:
        """Release the webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Webcam released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class FPSCounter:
    """
    Simple FPS counter using exponential moving average.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Smoothing factor for EMA (lower = smoother).
        """
        self.alpha = alpha
        self._fps = 0.0
        self._last_time = time.perf_counter()

    def update(self) -> float:
        """
        Update the FPS counter and return current FPS.

        Returns:
            Current smoothed FPS.
        """
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = self.alpha * instant_fps + (1 - self.alpha) * self._fps

        return self._fps

    @property
    def fps(self) -> float:
        """Current FPS value."""
        return self._fps
