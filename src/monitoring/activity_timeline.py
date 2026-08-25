"""
Activity Timeline Module (Stage 3)

Collects scene-level action predictions that occur while a person session is
active.  Uses the smoothed action from Stage 2 to avoid micro-segments from
single-frame prediction noise.

IMPORTANT: These are *scene-level* activities observed during the person's
presence — NOT actions performed by the identified person.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from src.action.config import get_status
from src.monitoring.config import ACTIVITY_MIN_SEGMENT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class ActivitySegment:
    """One contiguous period where a particular scene action was observed."""

    action: str
    start_time: float      # time.time() epoch
    end_time: float        # time.time() epoch
    status: str            # "NORMAL" or "SUSPICIOUS"

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict:
        from datetime import datetime, timezone

        return {
            "action": self.action,
            "start_time": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat(),
            "status": self.status,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class ActivityTimeline:
    """
    Maintains a timeline of scene activities observed during a person's
    monitoring session.

    Each call to ``update()`` provides the current *smoothed* scene action.
    Consecutive frames with the same action are merged into a single segment.
    """

    def __init__(self, min_segment_seconds: float = ACTIVITY_MIN_SEGMENT_SECONDS):
        self.min_segment_seconds = min_segment_seconds
        self._segments: List[ActivitySegment] = []
        self._current_action: Optional[str] = None
        self._current_start: Optional[float] = None

    def update(self, action: str, now: Optional[float] = None) -> None:
        """
        Feed the latest smoothed scene action.

        If the action changes, the previous segment is finalised (if it met the
        minimum duration) and a new segment starts.

        Args:
            action: Current smoothed action name (e.g. "Walk").
            now:    Current timestamp (defaults to ``time.time()``).
        """
        if now is None:
            now = time.time()

        if action == self._current_action:
            # Same action continues — nothing to do
            return

        # Action changed — finalise the previous segment
        self._finalise_current(now)

        # Start new segment
        self._current_action = action
        self._current_start = now

    def finalize(self, now: Optional[float] = None) -> None:
        """
        Called when the person session ends (exit detected or app shutdown).

        Closes whatever segment is currently open.
        """
        if now is None:
            now = time.time()
        self._finalise_current(now)

    def _finalise_current(self, now: float) -> None:
        """Append the current segment if it met the minimum duration."""
        if self._current_action is None or self._current_start is None:
            return

        duration = now - self._current_start
        if duration >= self.min_segment_seconds:
            segment = ActivitySegment(
                action=self._current_action,
                start_time=self._current_start,
                end_time=now,
                status=get_status(self._current_action),
            )
            self._segments.append(segment)

        self._current_action = None
        self._current_start = None

    @property
    def segments(self) -> List[ActivitySegment]:
        """Return the list of finalised segments."""
        return list(self._segments)

    @property
    def unique_actions(self) -> List[str]:
        """Return deduplicated list of actions observed (in order of first appearance)."""
        seen = set()
        result = []
        for seg in self._segments:
            if seg.action not in seen:
                seen.add(seg.action)
                result.append(seg.action)
        return result

    @property
    def suspicious_actions(self) -> List[str]:
        """Return deduplicated list of suspicious actions observed."""
        seen = set()
        result = []
        for seg in self._segments:
            if seg.status == "SUSPICIOUS" and seg.action not in seen:
                seen.add(seg.action)
                result.append(seg.action)
        return result

    def to_dicts(self) -> List[dict]:
        """Serialise all segments to a list of dicts."""
        return [seg.to_dict() for seg in self._segments]

    def reset(self) -> None:
        """Clear the timeline."""
        self._segments.clear()
        self._current_action = None
        self._current_start = None
