"""
Stage 3 — Monitoring Configuration

Centralized configuration for person tracking, session management,
identity association, exit grace periods, and monitoring log storage.
"""

from pathlib import Path

# ── Project Root ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── BoT-SORT Tracking ─────────────────────────────────────────────────────
# Which tracker YAML to use (ultralytics ships both bytetrack.yaml and botsort.yaml)
TRACKER_CONFIG = "botsort.yaml"

# Minimum detection confidence to feed into the tracker
TRACK_MIN_CONFIDENCE = 0.5

# ── Session Lifecycle ──────────────────────────────────────────────────────

# How many consecutive frames a track must be MISSING before we consider the
# person exited.  This acts as an occlusion / temporary-miss grace period.
# At ~30 FPS a value of 30 ≈ 1 second of grace.
EXIT_GRACE_FRAMES = 30

# A newly-confirmed track must appear for at least this many frames before
# we create a monitoring session — prevents flicker-driven false entries.
ENTRY_CONFIRM_FRAMES = 3

# ── Identity Association ──────────────────────────────────────────────────

# How often (in frames) to refresh/re-evaluate identity for an active track
IDENTITY_REFRESH_INTERVAL = 5

# Once a track receives a confident identity match (similarity >= this),
# it is "locked in" and will not revert to UNKNOWN on a single missed frame.
IDENTITY_LOCK_THRESHOLD = 0.50

# If the identity was locked but we have N consecutive frames without a
# face or with a low-confidence match, allow the identity to be re-evaluated
# instead of keeping a stale one.
IDENTITY_UNLOCK_FRAMES = 60

# ── Activity Timeline ─────────────────────────────────────────────────────

# Minimum duration (in seconds) for a new action segment to be appended to
# the activity timeline.  Prevents micro-segments from noise.
ACTIVITY_MIN_SEGMENT_SECONDS = 1.0

# ── Monitoring Logs ────────────────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
