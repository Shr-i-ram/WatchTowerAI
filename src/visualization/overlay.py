"""
Visualization Overlay Module

Draws person bounding boxes, identity labels, confidence scores,
and FPS information on the video frame.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.detection.person_detector import PersonDetection
from src.face.gallery import GalleryMatch

logger = logging.getLogger(__name__)


# ── Color Palette ──────────────────────────────────────────────────────────

COLOR_KNOWN = (0, 200, 0)        # Green for known identity
COLOR_UNKNOWN = (0, 0, 255)      # Red for unknown
COLOR_PERSON_BOX = (255, 180, 0) # Blue-ish for person bounding box
COLOR_FPS = (255, 255, 255)      # White for FPS text
COLOR_BG = (0, 0, 0)             # Black background for labels


@dataclass
class TemporalSmoothingState:
    """
    Tracks recent identity predictions for temporal smoothing.

    Maintains a short history window and returns the most frequent
    identity within that window to prevent label flickering.
    """

    window_size: int = 5  # Number of recent frames to consider
    min_confidence_frames: int = 3  # Min frames before switching identity

    _history: List[Tuple[str, float]] = field(default_factory=list)
    _stable_identity: str = ""
    _stable_similarity: float = 0.0

    def update(self, identity: str, similarity: float) -> Tuple[str, float]:
        """
        Add a new prediction and return the smoothed identity.

        Args:
            identity: Current frame's predicted identity.
            similarity: Current frame's similarity score.

        Returns:
            Tuple of (smoothed_identity, smoothed_similarity).
        """
        self._history.append((identity, similarity))

        # Trim to window
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]

        # Count identity occurrences in history
        identity_counts: Dict[str, int] = {}
        identity_sims: Dict[str, List[float]] = {}
        for ident, sim in self._history:
            identity_counts[ident] = identity_counts.get(ident, 0) + 1
            identity_sims.setdefault(ident, []).append(sim)

        # Find the most frequent identity
        most_frequent = max(identity_counts, key=lambda k: identity_counts[k])
        most_freq_count = identity_counts[most_frequent]

        # Calculate average similarity for the most frequent identity
        avg_sim = sum(identity_sims[most_frequent]) / len(identity_sims[most_frequent])

        # Only switch if the most frequent identity appears enough times
        if most_freq_count >= self.min_confidence_frames or len(self._history) < self.window_size:
            self._stable_identity = most_frequent
            self._stable_similarity = avg_sim
        # If we have enough history but no consensus, keep previous stable identity
        # This prevents flickering to UNKNOWN when a known person briefly loses detection

        return self._stable_identity, self._stable_similarity

    def reset(self) -> None:
        """Reset the smoothing state."""
        self._history.clear()
        self._stable_identity = ""
        self._stable_similarity = 0.0


class OverlayRenderer:
    """
    Renders visual overlays on the video frame.

    Handles person bounding boxes, identity labels, confidence scores,
    and FPS display with temporal smoothing.
    """

    def __init__(
        self,
        font_scale: float = 0.7,
        font_thickness: int = 2,
        box_thickness: int = 2,
        label_height: int = 30,
    ):
        """
        Args:
            font_scale: Scale for OpenCV putText.
            font_thickness: Thickness for OpenCV putText.
            box_thickness: Thickness for person bounding boxes.
            label_height: Height of the label background rectangle.
        """
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.box_thickness = box_thickness
        self.label_height = label_height

    def render(
        self,
        frame: np.ndarray,
        persons: List[PersonDetection],
        matches: Dict[int, GalleryMatch],  # person_index -> match
        fps: float,
        smoothing_states: Dict[int, TemporalSmoothingState],
        window_name: str = "L&T CCTV AI - Stage 1",
    ) -> np.ndarray:
        """
        Render all overlays on the frame.

        Args:
            frame: BGR image to annotate (will be modified in-place).
            persons: List of detected persons.
            matches: Dict mapping person index to GalleryMatch.
            fps: Current FPS value.
            smoothing_states: Dict mapping person index to smoothing state.
            window_name: Title for the cv2 window.

        Returns:
            The annotated frame.
        """
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # ── Draw person bounding boxes and labels ──────────────────────────
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person.bbox.astype(int)

            # Draw person bounding box
            color = COLOR_PERSON_BOX
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, self.box_thickness)

            # Get match for this person
            match = matches.get(i)
            if match is None:
                label = "NO FACE"
                display_sim = 0.0
                label_color = (128, 128, 128)  # Gray
            else:
                # Apply temporal smoothing
                smoothing = smoothing_states.get(i)
                if smoothing is not None:
                    smoothed_id, smoothed_sim = smoothing.update(
                        match.identity, match.similarity
                    )
                else:
                    smoothed_id = match.identity
                    smoothed_sim = match.similarity

                if smoothed_id == "UNKNOWN":
                    label = f"UNKNOWN | {smoothed_sim * 100:.1f}%"
                    label_color = COLOR_UNKNOWN
                else:
                    label = f"{smoothed_id} | {smoothed_sim * 100:.1f}%"
                    label_color = COLOR_KNOWN

                display_sim = smoothed_sim

            # ── Draw label background ─────────────────────────────────────
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
            )

            label_x1 = x1
            label_y1 = y1 - self.label_height - baseline
            label_x2 = x1 + text_w + 10
            label_y2 = y1 - baseline

            # Ensure label stays within frame
            if label_y1 < 0:
                label_y1 = y1 + baseline + 5
                label_y2 = y1 + baseline + self.label_height + 5

            # Draw filled rectangle background
            cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), _COLOR_BG, -1)

            # Draw label text
            cv2.putText(
                overlay,
                label,
                (label_x1 + 5, label_y1 + self.label_height - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                label_color,
                self.font_thickness,
                cv2.LINE_AA,
            )

        # ── Draw FPS and stats ────────────────────────────────────────────
        stats_text = f"FPS: {fps:.1f}  |  Persons: {len(persons)}"
        (tw, th), _ = cv2.getTextSize(
            stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )

        # Background for stats
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + tw + 10, 10 + th + 10),
            _COLOR_BG,
            -1,
        )
        cv2.putText(
            overlay,
            stats_text,
            (15, 10 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_FPS,
            1,
            cv2.LINE_AA,
        )

        return overlay


# Module-level constant to avoid name conflicts
_COLOR_BG = (0, 0, 0)
