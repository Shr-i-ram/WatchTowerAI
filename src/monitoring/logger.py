"""
Monitoring Logger Module (Stage 3)

Saves completed person sessions to structured JSON logs.
Each webcam run generates a timestamped log file so sessions are never
overwritten across application runs.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from src.monitoring.config import LOG_DIR
from src.monitoring.session import PersonSession

logger = logging.getLogger(__name__)


class MonitoringLogger:
    """
    Writes monitoring session data to structured JSON logs.

    Each application run creates a new log file named with a timestamp,
    e.g. ``logs/session_20260825_103104.json``.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file: Optional[Path] = None
        self._run_sessions: List[dict] = []
        self._initialise_run()

    def _initialise_run(self) -> None:
        """Create a new log file for this application run."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = self.log_dir / f"session_{ts}.json"
        self._run_sessions = []
        logger.info(f"Monitoring log: {self._log_file}")

    def save_session(self, session: PersonSession) -> None:
        """
        Append a completed session to the current run's log.

        The file is written incrementally so data is preserved even if
        the application crashes.
        """
        session_dict = session.to_dict()
        self._run_sessions.append(session_dict)
        self._flush()

    def save_sessions_bulk(self, sessions: List[PersonSession]) -> None:
        """Append multiple completed sessions and flush."""
        for s in sessions:
            self._run_sessions.append(s.to_dict())
        self._flush()

    def _flush(self) -> None:
        """Write the accumulated sessions to disk."""
        if self._log_file is None:
            return

        data = {
            "log_file": str(self._log_file.name),
            "total_sessions": len(self._run_sessions),
            "sessions": self._run_sessions,
        }

        try:
            with open(self._log_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write monitoring log: {e}")

    def finalize(self, active_sessions: List[PersonSession]) -> None:
        """
        Called on application shutdown.

        Finalizes any still-active sessions and saves everything.
        """
        for session in active_sessions:
            session.finalize()
            self._run_sessions.append(session.to_dict())

        self._flush()

        if self._log_file:
            logger.info(
                f"Monitoring log saved: {self._log_file} "
                f"({len(self._run_sessions)} sessions)"
            )

    @property
    def log_path(self) -> Optional[Path]:
        return self._log_file

    @property
    def session_count(self) -> int:
        return len(self._run_sessions)
