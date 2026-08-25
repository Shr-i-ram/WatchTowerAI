"""
Person Tracker Module (Stage 3)

Wraps ultralytics' built-in BoT-SORT tracker to assign persistent track IDs
to detected persons across frames.

Uses the same YOLO model already loaded for detection — no duplicate inference.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from src.detection.person_detector import PersonDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    """A person detection enriched with a persistent track ID."""

    bbox: np.ndarray       # [x1, y1, x2, y2]
    confidence: float
    track_id: int          # Persistent ID assigned by BoT-SORT


class PersonTracker:
    """
    BoT-SORT person tracker built on top of ultralytics.

    The tracker receives the same YOLO model instance used for detection,
    so detection inference is only performed once per frame.
    """

    PERSON_CLASS_ID = 0  # COCO "person"

    def __init__(self, tracker_config: str = "botsort.yaml"):
        """
        Args:
            tracker_config: Name of the ultralytics tracker YAML to use.
        """
        self.tracker_config = tracker_config
        self._tracker = None

    def init_tracker(self) -> None:
        """Create the internal BoT-SORT tracker object."""
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.utils import get_tracker_config

        cfg_path = get_tracker_config(self.tracker_config)
        self._tracker = BOTSORT(args=cfg_path)
        logger.info(f"BoT-SORT tracker initialised (config={self.tracker_config})")

    def update(
        self,
        yolo_model: YOLO,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> List[TrackedPerson]:
        """
        Run detection + tracking on a single frame and return tracked persons.

        This calls ``yolo_model.track()`` which internally runs YOLO detection
        and then feeds the detections into BoT-SORT for association across frames.

        Args:
            yolo_model: Loaded ultralytics YOLO model instance.
            frame: BGR image (H, W, 3).
            confidence_threshold: Minimum confidence to keep a detection.

        Returns:
            List of TrackedPerson objects with persistent IDs.
        """
        results = yolo_model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=confidence_threshold,
            iou=0.45,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        tracked: List[TrackedPerson] = []

        if result.boxes is None:
            return tracked

        boxes = result.boxes

        # Extract track IDs — if tracker is active, boxes.id will be set
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):
            tid = int(track_ids[i]) if track_ids is not None else -1
            tracked.append(
                TrackedPerson(
                    bbox=xyxy[i].astype(np.int32),
                    confidence=float(confs[i]),
                    track_id=tid,
                )
            )

        return tracked

    def reset(self) -> None:
        """Reset the tracker state (e.g. when restarting the webcam)."""
        if self._tracker is not None:
            self._tracker.reset()
            logger.info("Tracker state reset.")
