"""
Person Detection Module

YOLO-based person detection using ultralytics.
Detects people in video frames and returns bounding boxes.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """A detected person with bounding box and confidence."""

    bbox: np.ndarray  # [x1, y1, x2, y2] in pixel coordinates
    confidence: float  # detection confidence score

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> tuple:
        cx = int((self.bbox[0] + self.bbox[2]) / 2)
        cy = int((self.bbox[1] + self.bbox[3]) / 2)
        return (cx, cy)

    def contains_point(self, x: int, y: int, margin: float = 0.1) -> bool:
        """Check if a point is inside this bbox with optional margin."""
        x1, y1, x2, y2 = self.bbox
        mx = margin * (x2 - x1)
        my = margin * (y2 - y1)
        return (x1 - mx) <= x <= (x2 + mx) and (y1 - my) <= y <= (y2 + my)

    def intersection_over_union(self, other: "PersonDetection") -> float:
        """Calculate IoU with another detection."""
        x1 = max(self.bbox[0], other.bbox[0])
        y1 = max(self.bbox[1], other.bbox[1])
        x2 = min(self.bbox[2], other.bbox[2])
        y2 = min(self.bbox[3], other.bbox[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_self = self.width * self.height
        area_other = other.width * other.height
        union = area_self + area_other - intersection

        return intersection / union if union > 0 else 0.0


class PersonDetector:
    """
    YOLO-based person detector.

    Uses YOLO11m (or fallback to YOLOv8m) for detecting people in frames.
    Only the 'person' class (COCO class 0) is retained.
    """

    # COCO class index for "person"
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolo11m.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """
        Initialize the person detector.

        Args:
            model_name: YOLO model name/path. Will auto-download if not local.
            confidence_threshold: Minimum confidence to keep a detection.
            iou_threshold: NMS IoU threshold for suppression.
            device: Inference device ('cuda', 'cpu', or None for auto).
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model_name = model_name
        self.device = device
        self.model = None

    def load(self) -> str:
        """
        Load the YOLO model onto the specified device.

        Returns:
            Device string where model is loaded.
        """
        from ultralytics import YOLO

        logger.info(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)

        # Determine device
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"YOLO model loaded on device: {self.device}")
        return self.device

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Detect all people in a video frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of PersonDetection objects.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],  # Only detect people
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                # box.xyxy[0] gives [x1, y1, x2, y2]
                xyxy = box.xyxy[0].cpu().numpy().astype(np.int32)
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    PersonDetection(bbox=xyxy, confidence=conf)
                )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return detections
