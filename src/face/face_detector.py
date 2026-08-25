"""
Face Detection Module

Uses InsightFace's SCRFD model for face detection and alignment.
Operates within person bounding boxes to find faces belonging to detected people.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """A detected face with bounding box, landmarks, and alignment info."""

    bbox: np.ndarray  # [x1, y1, x2, y2] in full-frame pixel coordinates
    kps: np.ndarray  # (5, 2) facial keypoints [left_eye, right_eye, nose, left_mouth, right_mouth]
    confidence: float  # detection confidence
    aligned_face: Optional[np.ndarray] = None  # 112x112 aligned face crop (set later)


class FaceDetector:
    """
    SCRFD-based face detector using InsightFace.

    Detects and aligns faces within a given image region.
    The face detector operates on the full frame but can be restricted
    to a sub-region (person bounding box) for efficiency.
    """

    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the face detector.

        Args:
            det_size: Input size for the SCRFD detector (width, height).
            confidence_threshold: Minimum face detection confidence.
        """
        self.det_size = det_size
        self.confidence_threshold = confidence_threshold
        self._analysis = None  # InsightFace FaceAnalysis

    def load(self) -> str:
        """
        Load the SCRFD face detection model.

        Returns:
            Device string where model is loaded.
        """
        import insightface
        import onnxruntime as ort

        # Determine providers
        providers = []
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            device = "cuda"
        else:
            device = "cpu"
        providers.append("CPUExecutionProvider")

        logger.info(f"Loading InsightFace FaceAnalysis (SCRFD) on providers: {providers}")

        self._analysis = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self._analysis.prepare(ctx_id=0 if device == "cuda" else -1, det_size=self.det_size)

        logger.info(f"Face detector loaded. Detection size: {self.det_size}")
        return device

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect all faces in a full frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of FaceDetection objects.
        """
        if self._analysis is None:
            raise RuntimeError("Face detector not loaded. Call load() first.")

        # InsightFace expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self._analysis.get(rgb)

        detections = []
        for face in faces:
            if face.det_score < self.confidence_threshold:
                continue
            det = FaceDetection(
                bbox=face.bbox.astype(np.int32),
                kps=face.kps.astype(np.float32),
                confidence=float(face.det_score),
                aligned_face=face.normed_embedding if hasattr(face, "normed_embedding") else None,
            )
            detections.append(det)

        return detections

    def detect_in_region(
        self, frame: np.ndarray, region_bbox: np.ndarray
    ) -> List[FaceDetection]:
        """
        Detect faces within a specific region (e.g., a person bounding box).

        This crops the region, runs face detection on it, then maps
        the coordinates back to the full frame.

        Args:
            frame: Full BGR image (H, W, 3).
            region_bbox: Region [x1, y1, x2, y2] to search within.

        Returns:
            List of FaceDetection objects with coordinates in full frame.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region_bbox.astype(int)

        # Clamp to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return []

        # Crop the region
        crop = frame[y1:y2, x1:x2]

        # Detect faces in the crop
        crop_faces = self.detect(crop)

        # Map coordinates back to full frame
        detections = []
        for face in crop_faces:
            # Offset bbox
            mapped_bbox = face.bbox.copy()
            mapped_bbox[0] += x1
            mapped_bbox[2] += x1
            mapped_bbox[1] += y1
            mapped_bbox[3] += y1

            # Offset keypoints
            mapped_kps = face.kps.copy()
            mapped_kps[:, 0] += x1
            mapped_kps[:, 1] += y1

            detections.append(
                FaceDetection(
                    bbox=mapped_bbox,
                    kps=mapped_kps,
                    confidence=face.confidence,
                    aligned_face=face.aligned_face,
                )
            )

        return detections
