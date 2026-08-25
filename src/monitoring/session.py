"""
Person Session Module (Stage 3)

Manages the lifecycle of a tracked person's monitoring session:
  - Entry detection (with confirmation grace)
  - Identity association and stability
  - Activity timeline collection
  - Exit detection (with grace period for temporary occlusion)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.monitoring.activity_timeline import ActivityTimeline
from src.monitoring.config import (
    ENTRY_CONFIRM_FRAMES,
    EXIT_GRACE_FRAMES,
    IDENTITY_LOCK_THRESHOLD,
    IDENTITY_UNLOCK_FRAMES,
)

logger = logging.getLogger(__name__)


@dataclass
class PersonSession:
    """One continuous monitoring session for a tracked person."""

    track_id: int
    identity: str = "UNKNOWN"
    entry_time: Optional[float] = None
    exit_time: Optional[float] = None
    last_seen_frame: int = 0
    activity_timeline: ActivityTimeline = field(default_factory=ActivityTimeline)

    # Identity stability
    _best_identity: str = "UNKNOWN"
    _best_similarity: float = 0.0
    _frames_since_face: int = 0
    _identity_locked: bool = False
    _frames_seen: int = 0

    def __post_init__(self):
        if self.entry_time is None:
            self.entry_time = time.time()

    @property
    def duration_seconds(self) -> float:
        end = self.exit_time if self.exit_time is not None else time.time()
        return end - self.entry_time

    @property
    def is_active(self) -> bool:
        return self.exit_time is None

    def update_identity(
        self,
        identity: str,
        similarity: float,
        frame_num: int,
    ) -> None:
        """
        Update the identity for this session based on a face recognition result.

        Once a high-confidence identity is established, it is locked in and will
        not be reverted to UNKNOWN on a single missed face.
        """
        self.last_seen_frame = frame_num
        self._frames_seen += 1

        if identity != "UNKNOWN" and similarity >= IDENTITY_LOCK_THRESHOLD:
            # Strong match — update best and lock
            if similarity > self._best_similarity:
                self._best_identity = identity
                self._best_similarity = similarity
                self.identity = identity
                self._identity_locked = True
                self._frames_since_face = 0
                logger.debug(
                    f"Track #{self.track_id}: identity locked → {identity} "
                    f"(sim={similarity:.3f})"
                )
        elif identity == "UNKNOWN":
            self._frames_since_face += 1

            # If identity is locked, allow some frames without a face before
            # considering unlocking (person may have turned away).
            if self._identity_locked and self._frames_since_face > IDENTITY_UNLOCK_FRAMES:
                self._identity_locked = False
                self.identity = "UNKNOWN"
                logger.debug(
                    f"Track #{self.track_id}: identity unlocked after "
                    f"{self._frames_since_face} frames without face"
                )
            # If not locked and we just started, remain UNKNOWN
            elif not self._identity_locked:
                pass  # stay UNKNOWN
        else:
            # Weak match (below lock threshold) — only use if no better identity yet
            if not self._identity_locked:
                if similarity > self._best_similarity:
                    self._best_identity = identity
                    self._best_similarity = similarity
                    self.identity = identity

    def mark_seen(self, frame_num: int) -> None:
        """Mark that this person was detected this frame (without identity info)."""
        self.last_seen_frame = frame_num
        self._frames_seen += 1

    def finalize(self, now: Optional[float] = None) -> None:
        """Finalize the session: record exit time and close the activity timeline."""
        if now is None:
            now = time.time()
        self.exit_time = now
        self.activity_timeline.finalize(now)
        logger.info(
            f"Session finalized: Track #{self.track_id}, "
            f"Identity={self.identity}, "
            f"Duration={self.duration_seconds:.1f}s, "
            f"Activities={self.activity_timeline.unique_actions}"
        )

    def to_dict(self) -> dict:
        """Serialise the session to a dict suitable for JSON storage."""
        from datetime import datetime, timezone

        def _fmt(ts):
            if ts is None:
                return None
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        return {
            "identity": self.identity,
            "track_id": self.track_id,
            "entry_time": _fmt(self.entry_time),
            "exit_time": _fmt(self.exit_time),
            "duration_seconds": round(self.duration_seconds, 2),
            "scene_activity_timeline": self.activity_timeline.to_dicts(),
            "scene_actions_observed": self.activity_timeline.unique_actions,
            "suspicious_actions_observed": self.activity_timeline.suspicious_actions,
        }


class SessionManager:
    """
    Manages all active person sessions and handles entry/exit lifecycle.

    For each tracked person (identified by track_id), the manager:
    - Creates a new session when a track is confirmed (appears for N frames)
    - Updates identity as face recognition results arrive
    - Records scene activities via the activity timeline
    - Finalises the session when a track disappears (grace period)
    """

    def __init__(
        self,
        entry_confirm_frames: int = ENTRY_CONFIRM_FRAMES,
        exit_grace_frames: int = EXIT_GRACE_FRAMES,
    ):
        self.entry_confirm_frames = entry_confirm_frames
        self.exit_grace_frames = exit_grace_frames

        # Active sessions keyed by track_id
        self._active: Dict[int, PersonSession] = {}
        # Pending tracks (seen but not yet confirmed)
        self._pending: Dict[int, int] = {}  # track_id → consecutive frame count
        # Completed sessions (for log saving)
        self._completed: List[PersonSession] = []

        self._frame_count: int = 0

    def update(
        self,
        tracked_track_ids: set,
        frame_num: int,
        current_scene_action: str = "Stand",
    ) -> None:
        """
        Update all sessions based on the current frame's tracked IDs.

        Args:
            tracked_track_ids: Set of track IDs visible in the current frame.
            frame_num: Current frame number.
            current_scene_action: Current smoothed scene action from Stage 2.
        """
        self._frame_count = frame_num

        # ── Handle active sessions ────────────────────────────────────────
        for tid, session in list(self._active.items()):
            if tid in tracked_track_ids:
                # Person is visible — keep session alive
                session.mark_seen(frame_num)
                session.activity_timeline.update(current_scene_action)
            else:
                # Person not seen this frame — check grace period
                frames_missing = frame_num - session.last_seen_frame
                if frames_missing >= self.exit_grace_frames:
                    # Grace period expired → finalize
                    session.finalize()
                    self._completed.append(session)
                    del self._active[tid]
                    logger.info(
                        f"[EXIT] Track #{tid} | Identity: {session.identity} | "
                        f"Duration: {session.duration_seconds:.1f}s"
                    )

        # ── Handle pending tracks ─────────────────────────────────────────
        for tid in list(self._pending.keys()):
            if tid in tracked_track_ids:
                self._pending[tid] += 1
                if self._pending[tid] >= self.entry_confirm_frames:
                    # Confirmed — create a new session
                    session = PersonSession(track_id=tid, entry_time=time.time())
                    self._active[tid] = session
                    del self._pending[tid]
                    logger.info(
                        f"[ENTRY] Track #{tid} | Identity: UNKNOWN"
                    )
            else:
                # Track disappeared before confirmation — discard
                del self._pending[tid]

        # ── Detect new tracks ─────────────────────────────────────────────
        for tid in tracked_track_ids:
            if tid not in self._active and tid not in self._pending:
                self._pending[tid] = 1

    def update_identity(
        self,
        track_id: int,
        identity: str,
        similarity: float,
        frame_num: int,
    ) -> None:
        """
        Update the identity for an active session.

        Called when face recognition produces a result for a tracked person.

        Args:
            track_id: The track's persistent ID.
            identity: Recognised identity name or "UNKNOWN".
            similarity: Cosine similarity score.
            frame_num: Current frame number.
        """
        session = self._active.get(track_id)
        if session is not None:
            session.update_identity(identity, similarity, frame_num)

    def finalize_all(self) -> None:
        """Finalize all active and pending sessions (called on app shutdown)."""
        now = time.time()
        for session in self._active.values():
            session.finalize(now)
            self._completed.append(session)
        self._active.clear()

        # Pending tracks that never became sessions are simply discarded
        self._pending.clear()

    @property
    def active_sessions(self) -> Dict[int, PersonSession]:
        """Currently active person sessions."""
        return dict(self._active)

    @property
    def completed_sessions(self) -> List[PersonSession]:
        """Sessions that have been finalized."""
        return list(self._completed)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def total_completed(self) -> int:
        return len(self._completed)
