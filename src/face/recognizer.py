"""
Face Recognizer Module

Uses ArcFace (via InsightFace) to generate face embeddings.
Embeddings are normalized 512-dimensional vectors used for identity matching.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.face.face_detector import FaceDetection

logger = logging.getLogger(__name__)


@dataclass
class RecognizedFace:
    """A face with its ArcFace embedding."""

    detection: FaceDetection
    embedding: np.ndarray  # (512,) L2-normalized ArcFace embedding


class FaceRecognizer:
    """
    ArcFace-based face recognizer using InsightFace.

    Takes detected faces (with alignment) and generates embeddings
    for gallery comparison.
    """

    EMBEDDING_DIM = 512

    def __init__(self):
        """Initialize the face recognizer."""
        self._analysis = None  # InsightFace FaceAnalysis (shared with detector or standalone)

    def load(self) -> str:
        """
        Load the ArcFace model via InsightFace.

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

        logger.info(f"Loading InsightFace ArcFace model on providers: {providers}")

        self._analysis = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        self._analysis.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        logger.info("ArcFace model loaded successfully.")
        return device

    def load_from_shared(self, face_detector_analysis):
        """
        Share the InsightFace FaceAnalysis instance with the face detector.

        This avoids loading the model twice. The FaceAnalysis object handles
        both detection (SCRFD) and recognition (ArcFace).

        Args:
            face_detector_analysis: InsightFace FaceAnalysis instance from FaceDetector.
        """
        self._analysis = face_detector_analysis
        logger.info("ArcFace model loaded via shared FaceAnalysis instance.")

    def generate_embedding(self, frame: np.ndarray, face: FaceDetection) -> Optional[np.ndarray]:
        """
        Generate an ArcFace embedding for a single detected face.

        Uses the pre-computed aligned face from InsightFace if available,
        otherwise falls back to cropping and aligning manually.

        Args:
            frame: Full BGR image (H, W, 3).
            face: FaceDetection with bbox and keypoints.

        Returns:
            L2-normalized (512,) embedding or None if alignment fails.
        """
        if self._analysis is None:
            raise RuntimeError("Recognizer not loaded. Call load() or load_from_shared().")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get the face from InsightFace which includes embedding
        faces = self._analysis.get(rgb)

        # Find the face that best overlaps with our detection
        best_match = None
        best_iou = 0.0

        for detected_face in faces:
            if detected_face.det_score < 0.5:
                continue

            # Calculate IoU between detected face and our detection
            iou = self._compute_iou(detected_face.bbox.astype(int), face.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = detected_face

        if best_match is not None and hasattr(best_match, "normed_embedding"):
            return best_match.normed_embedding.astype(np.float32)

        # Fallback: extract embedding from the aligned face if available
        if face.aligned_face is not None:
            return face.aligned_face

        logger.warning("Could not generate embedding for face detection.")
        return None

    def generate_embeddings_batch(
        self, frame: np.ndarray, faces: List[FaceDetection]
    ) -> List[RecognizedFace]:
        """
        Generate embeddings for multiple detected faces in a frame.

        This is more efficient than calling generate_embedding() per face,
        since InsightFace processes all faces in a single forward pass.

        Args:
            frame: Full BGR image (H, W, 3).
            faces: List of FaceDetection objects.

        Returns:
            List of RecognizedFace objects (only those with valid embeddings).
        """
        if self._analysis is None:
            raise RuntimeError("Recognizer not loaded. Call load() or load_from_shared().")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = self._analysis.get(rgb)

        results = []
        used_indices = set()

        for face_det in faces:
            best_match_idx = None
            best_iou = 0.0

            for i, detected in enumerate(detected_faces):
                if i in used_indices:
                    continue
                if detected.det_score < 0.3:
                    continue

                iou = self._compute_iou(detected.bbox.astype(int), face_det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i

            if best_match_idx is not None and best_iou > 0.3:
                used_indices.add(best_match_idx)
                detected = detected_faces[best_match_idx]
                if hasattr(detected, "normed_embedding") and detected.normed_embedding is not None:
                    results.append(
                        RecognizedFace(
                            detection=face_det,
                            embedding=detected.normed_embedding.astype(np.float32),
                        )
                    )

        return results

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute Intersection over Union between two bboxes [x1, y1, x2, y2]."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(1, (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        area2 = max(1, (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
