#!/usr/bin/env python3
"""
L&T CCTV AI - Integrated Webcam Demo (Stage 1 + 2 + 3)

Real-time webcam feed with:
  Stage 1: Person detection + face-based identity recognition
  Stage 2: Scene-level action recognition (13 actions → NORMAL/SUSPICIOUS)
  Stage 3: BoT-SORT person tracking, entry/exit monitoring & activity logging

Usage:
    python scripts/webcam_demo.py

Press 'q' to quit, 'r' to reload identity gallery.
"""

import sys
import os
import time
import logging
from pathlib import Path
from collections import deque

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure onnxruntime can find CUDA DLLs bundled with PyTorch
# Must be done BEFORE any insightface/ultralytics imports
def _setup_cuda_path():
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib) and torch_lib not in os.environ.get('PATH', ''):
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
    except ImportError:
        pass
    try:
        import onnxruntime
        onnxruntime.preload_dlls()
    except (ImportError, AttributeError):
        pass
_setup_cuda_path()
del _setup_cuda_path

import cv2
import numpy as np
import torch

from src.detection.person_detector import PersonDetector
from src.face.face_detector import FaceDetector
from src.face.recognizer import FaceRecognizer
from src.face.gallery import IdentityGallery
from src.camera.webcam import WebcamCapture, FPSCounter
from src.visualization.overlay import OverlayRenderer, TemporalSmoothingState

# ── Stage 2: Action Recognition ───────────────────────────────────────────
from src.action.config import (
    ACTION_SMOOTHING_MIN,
    ACTION_SMOOTHING_WINDOW,
    MODEL_SAVE_DIR,
    NUM_FRAMES,
    WEBCAM_ACTION_INTERVAL,
    WEBCAM_BUFFER_SIZE,
    get_status,
)
from src.action.inference import ActionRecognizer, TemporalActionState

# ── Stage 3: Tracking & Monitoring ────────────────────────────────────────
from src.tracking.tracker import PersonTracker, TrackedPerson
from src.monitoring.session import SessionManager, PersonSession
from src.monitoring.logger import MonitoringLogger
from src.monitoring.config import (
    TRACK_MIN_CONFIDENCE,
    IDENTITY_REFRESH_INTERVAL,
)

# ── Configuration ──────────────────────────────────────────────────────────

# Paths
GALLERY_DIR = PROJECT_ROOT / "models" / "face_gallery"

# YOLO person detection
YOLO_MODEL = str(PROJECT_ROOT / "yolo11m.pt")
YOLO_CONFIDENCE = 0.5
YOLO_IOU = 0.45

# Recognition
SIMILARITY_THRESHOLD = 0.45

# Temporal smoothing
SMOOTHING_WINDOW = 5
SMOOTHING_MIN_FRAMES = 3

# Webcam
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ── Logging Setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device_info() -> str:
    """Print detailed device information."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            return f"CUDA (GPU: {device_name}, Devices: {device_count})"
        else:
            return "CPU (no CUDA available - GPU acceleration disabled)"
    except ImportError:
        return "CPU (PyTorch not installed)"


def print_banner():
    """Print application banner."""
    print()
    print("=" * 65)
    print("  L&T CCTV AI - Integrated Monitoring Prototype")
    print("  Stage 1: Person Identification")
    print("  Stage 2: Scene-Level Action Recognition")
    print("  Stage 3: Person Tracking & Monitoring")
    print("=" * 65)
    print()


def draw_bottom_panel(
    frame: np.ndarray,
    action: str,
    confidence: float,
    status: str,
    fps: float,
    num_persons: int,
    num_active_sessions: int,
) -> np.ndarray:
    """
    Draw the combined Stage 2 + Stage 3 information panel at the bottom.
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # ── Action info panel (bottom-left) ────────────────────────────────────
    panel_h = 140
    panel_w = 420
    panel_x = 10
    panel_y = h - panel_h - 10

    # Semi-transparent background
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (255, 255, 255), 1)

    # Title
    cv2.putText(
        overlay, "SCENE ACTION", (panel_x + 10, panel_y + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )

    # Action name
    cv2.putText(
        overlay, f"{action}", (panel_x + 10, panel_y + 48),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )

    # Confidence
    cv2.putText(
        overlay, f"Confidence: {confidence * 100:.1f}%", (panel_x + 10, panel_y + 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Status
    status_color = (0, 0, 255) if status == "SUSPICIOUS" else (0, 200, 0)
    status_text = f"STATUS: {status}"
    cv2.putText(
        overlay, status_text, (panel_x + 10, panel_y + 96),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA,
    )

    # Active sessions
    cv2.putText(
        overlay, f"Active Tracks: {num_active_sessions}", (panel_x + 10, panel_y + 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
    )

    # ── FPS and person count (top-left) ────────────────────────────────────
    stats_text = f"FPS: {fps:.1f}  |  Persons: {num_persons}"
    (tw, th), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(overlay, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
    cv2.putText(
        overlay, stats_text, (15, 10 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return overlay


def render_tracked_persons(
    frame: np.ndarray,
    tracked_persons: list,
    track_id_to_identity: dict,
    track_id_to_sim: dict,
) -> np.ndarray:
    """
    Draw person bounding boxes with identity + track ID labels.

    Args:
        frame: BGR image.
        tracked_persons: List of TrackedPerson from the tracker.
        track_id_to_identity: Dict[track_id → identity string].
        track_id_to_sim: Dict[track_id → similarity float].

    Returns:
        Annotated frame.
    """
    overlay = frame.copy()

    for person in tracked_persons:
        x1, y1, x2, y2 = person.bbox.astype(int)
        tid = person.track_id

        # Get identity info
        identity = track_id_to_identity.get(tid, "UNKNOWN")
        sim = track_id_to_sim.get(tid, 0.0)

        # Colors
        if identity == "UNKNOWN":
            label_color = (0, 0, 255)    # Red
            box_color = (0, 0, 255)
        else:
            label_color = (0, 200, 0)    # Green
            box_color = (0, 200, 0)

        # Draw bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

        # Build label: "Shriram | 96.4%" + "Track #7"
        if identity == "UNKNOWN":
            id_text = f"UNKNOWN | {sim * 100:.1f}%"
        else:
            id_text = f"{identity} | {sim * 100:.1f}%"
        track_text = f"Track #{tid}"

        # Draw identity label background + text
        font_scale = 0.65
        font_thickness = 2
        (tw, th), baseline = cv2.getTextSize(
            id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        label_x1 = x1
        label_y1 = y1 - 48 - baseline
        label_x2 = x1 + max(tw, 130) + 10
        label_y2 = y1 - baseline

        # Ensure label stays within frame
        if label_y1 < 0:
            label_y1 = y1 + baseline + 5
            label_y2 = y1 + baseline + 48 + 5

        # Background for identity
        cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
        cv2.putText(
            overlay, id_text,
            (label_x1 + 5, label_y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, font_thickness, cv2.LINE_AA,
        )

        # Background for track ID
        track_y1 = label_y2 + 2
        track_y2 = track_y2 = track_y1 + 22
        cv2.rectangle(overlay, (label_x1, track_y1), (label_x2, track_y2), (40, 40, 40), -1)
        cv2.putText(
            overlay, track_text,
            (label_x1 + 5, track_y1 + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA,
        )

    return overlay


def main():
    """Main integrated webcam inference loop."""
    print_banner()

    # ── Device Info ────────────────────────────────────────────────────────
    device_info = get_device_info()
    print(f"Device: {device_info}")
    print()

    # ── Load YOLO Person Detector (Stage 1) ────────────────────────────────
    print("[Stage 1] Loading YOLO person detector...")
    person_detector = PersonDetector(
        model_name=YOLO_MODEL,
        confidence_threshold=YOLO_CONFIDENCE,
        iou_threshold=YOLO_IOU,
    )
    detector_device = person_detector.load()
    print(f"  YOLO loaded on: {detector_device}")

    # ── Load Face Detector (Stage 1) ───────────────────────────────────────
    print("[Stage 1] Loading face detector (SCRFD)...")
    face_detector = FaceDetector()
    face_detector.load()

    # ── Load Face Recognizer (Stage 1) ─────────────────────────────────────
    print("[Stage 1] Loading face recognizer (ArcFace)...")
    face_recognizer = FaceRecognizer()
    face_recognizer.load_from_shared(face_detector._analysis)

    # ── Load Identity Gallery (Stage 1) ────────────────────────────────────
    print("[Stage 1] Loading identity gallery...")
    gallery = IdentityGallery(
        similarity_threshold=SIMILARITY_THRESHOLD,
        gallery_dir=str(GALLERY_DIR),
    )

    if not gallery.load():
        print()
        print("  WARNING: No gallery found. All persons will be UNKNOWN.")
        print(f"  Expected gallery at: {GALLERY_DIR}")
        print("  Run 'python scripts/build_gallery.py' to build the gallery.")
        print()

    # ── Load Action Recognition Model (Stage 2) ───────────────────────────
    print("[Stage 2] Loading action recognition model...")
    action_recognizer = ActionRecognizer()
    try:
        action_recognizer.load()
        stage2_loaded = True
        print(f"  Action model loaded on: {action_recognizer.device}")
    except FileNotFoundError:
        print()
        print("  WARNING: No trained action model found.")
        print(f"  Expected at: {MODEL_SAVE_DIR / 'best_model.pt'}")
        print("  Run 'python train_action.py' to train the model.")
        print("  Continuing with Stage 1 + Stage 3 only.")
        stage2_loaded = False
    except Exception as e:
        print(f"  WARNING: Could not load action model: {e}")
        print("  Continuing with Stage 1 + Stage 3 only.")
        stage2_loaded = False

    print()

    # ── Configuration Summary ──────────────────────────────────────────────
    print("Configuration:")
    print(f"  Identity threshold: {SIMILARITY_THRESHOLD}")
    print(f"  Smoothing window:   {SMOOTHING_WINDOW} frames")
    print(f"  Identity refresh:   every {IDENTITY_REFRESH_INTERVAL} frames")
    if stage2_loaded:
        print(f"  Action interval:    every {WEBCAM_ACTION_INTERVAL} frames")
        print(f"  Action buffer:      {WEBCAM_BUFFER_SIZE} frames")
    print()

    # ── Initialize Webcam ──────────────────────────────────────────────────
    print("Opening webcam...")
    webcam = WebcamCapture(
        camera_index=CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    if not webcam.open():
        print("ERROR: Could not open webcam. Exiting.")
        sys.exit(1)

    print("Webcam ready.")
    print()

    # ── Initialize Components ──────────────────────────────────────────────
    fps_counter = FPSCounter()
    frame_count = 0

    # Identity caching (keyed by track_id now, not list index)
    last_refresh: dict[int, int] = {}
    cached_matches: dict[int, object] = {}

    # Stage 3: Session manager
    session_mgr = SessionManager()
    monitor_log = MonitoringLogger()
    print(f"  Monitoring log: {monitor_log.log_path}")

    # Stage 2: Action recognition rolling buffer
    if stage2_loaded:
        action_buffer: deque = deque(maxlen=WEBCAM_BUFFER_SIZE)
        action_state = TemporalActionState()
        current_action = "Stand"
        current_action_conf = 0.0
        current_action_status = "NORMAL"

    print()
    print("Starting integrated demo. Press 'q' to quit, 'r' to reload gallery.")
    print()

    # ── Main Loop ──────────────────────────────────────────────────────────
    try:
        while True:
            # Read frame
            success, frame = webcam.read()
            if not success or frame is None:
                logger.warning("Failed to read frame. Retrying...")
                time.sleep(0.01)
                continue

            # Update FPS
            fps = fps_counter.update()
            frame_count += 1

            # ── Stage 1 + 3: Detection + Tracking ──────────────────────────
            # Use YOLO track() which runs detection + BoT-SORT in one pass
            results = person_detector.model.track(
                source=frame,
                persist=True,
                tracker="botsort.yaml",
                conf=YOLO_CONFIDENCE,
                iou=YOLO_IOU,
                classes=[0],  # person class only
                verbose=False,
            )

            tracked_persons: list[TrackedPerson] = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for i in range(len(xyxy)):
                    tid = int(track_ids[i]) if track_ids is not None else -1
                    tracked_persons.append(
                        TrackedPerson(
                            bbox=xyxy[i].astype(np.int32),
                            confidence=float(confs[i]),
                            track_id=tid,
                        )
                    )

            # Build set of visible track IDs
            visible_track_ids = {p.track_id for p in tracked_persons if p.track_id >= 0}

            # ── Face Recognition (keyed by track_id) ───────────────────────
            track_id_to_identity: dict[int, str] = {}
            track_id_to_sim: dict[int, float] = {}
            tracks_to_identify: list[TrackedPerson] = []
            track_faces: dict[int, list] = {}

            for person in tracked_persons:
                tid = person.track_id
                if tid < 0:
                    continue

                # Detect face within this person's bounding box
                faces_in_person = face_detector.detect_in_region(frame, person.bbox)
                track_faces[tid] = faces_in_person

                if not faces_in_person:
                    # No face — use cached identity if available
                    if tid in cached_matches:
                        m = cached_matches[tid]
                        track_id_to_identity[tid] = m.identity
                        track_id_to_sim[tid] = m.similarity
                    else:
                        track_id_to_identity[tid] = "UNKNOWN"
                        track_id_to_sim[tid] = 0.0
                    continue

                # Check if we should refresh identity
                should_refresh = (
                    tid not in last_refresh
                    or (frame_count - last_refresh[tid]) >= IDENTITY_REFRESH_INTERVAL
                    or tid not in cached_matches
                )

                if should_refresh:
                    tracks_to_identify.append(person)
                else:
                    m = cached_matches[tid]
                    track_id_to_identity[tid] = m.identity
                    track_id_to_sim[tid] = m.similarity

            # Batch face recognition for tracks that need it
            if tracks_to_identify and gallery.num_embeddings > 0:
                all_faces = []
                track_tid_map = []

                for person in tracks_to_identify:
                    tid = person.track_id
                    faces = track_faces.get(tid, [])
                    if faces:
                        all_faces.append(faces[0])
                        track_tid_map.append(tid)

                if all_faces:
                    recognized_faces = face_recognizer.generate_embeddings_batch(
                        frame, all_faces
                    )

                    for rf, tid in zip(recognized_faces, track_tid_map):
                        match_result = gallery.match(rf.embedding)
                        track_id_to_identity[tid] = match_result.identity
                        track_id_to_sim[tid] = match_result.similarity
                        last_refresh[tid] = frame_count
                        cached_matches[tid] = match_result

            # ── Stage 3: Update sessions ───────────────────────────────────
            session_mgr.update(
                tracked_track_ids=visible_track_ids,
                frame_num=frame_count,
                current_scene_action=current_action if stage2_loaded else "Stand",
            )

            # Update identities in active sessions
            for tid in visible_track_ids:
                identity = track_id_to_identity.get(tid, "UNKNOWN")
                sim = track_id_to_sim.get(tid, 0.0)
                if sim > 0:
                    session_mgr.update_identity(tid, identity, sim, frame_count)

            # Clean up caches for tracks that are no longer active
            stale_tids = set(last_refresh.keys()) - visible_track_ids
            for tid in stale_tids:
                last_refresh.pop(tid, None)
                cached_matches.pop(tid, None)

            # ── Stage 2: Action Recognition ────────────────────────────────
            if stage2_loaded:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                action_buffer.append(frame_rgb)

                if (
                    frame_count % WEBCAM_ACTION_INTERVAL == 0
                    and len(action_buffer) >= WEBCAM_BUFFER_SIZE
                ):
                    frames_arr = np.stack(list(action_buffer), axis=0)
                    try:
                        pred = action_recognizer.predict_video_tensor(frames_arr)
                        current_action, current_action_conf = action_state.update(
                            pred.action, pred.confidence
                        )
                        current_action_status = get_status(current_action)
                    except Exception as e:
                        logger.warning(f"Action prediction failed: {e}")

            # ── Render ─────────────────────────────────────────────────────
            annotated = render_tracked_persons(
                frame=frame,
                tracked_persons=tracked_persons,
                track_id_to_identity=track_id_to_identity,
                track_id_to_sim=track_id_to_sim,
            )

            # Draw bottom panel (action + tracking stats)
            annotated = draw_bottom_panel(
                annotated,
                action=current_action if stage2_loaded else "---",
                confidence=current_action_conf if stage2_loaded else 0.0,
                status=current_action_status if stage2_loaded else "NORMAL",
                fps=fps,
                num_persons=len(tracked_persons),
                num_active_sessions=session_mgr.active_count,
            )

            # Show frame
            cv2.imshow("L&T AI Monitoring Prototype", annotated)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuit signal received.")
                break
            elif key == ord("r"):
                print("\nReloading gallery...")
                gallery.load()
                cached_matches.clear()
                last_refresh.clear()
                print("Gallery reloaded.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # ── Graceful shutdown: finalize sessions and save logs ─────────────
        print("\nFinalizing monitoring sessions...")
        active = list(session_mgr.active_sessions.values())
        monitor_log.finalize(active)
        session_mgr.finalize_all()

        # Also save any completed sessions that weren't yet saved
        if session_mgr.completed_sessions:
            monitor_log.save_sessions_bulk(session_mgr.completed_sessions)

        print(f"  Sessions saved: {monitor_log.session_count}")
        print(f"  Log file: {monitor_log.log_path}")

        webcam.close()
        cv2.destroyAllWindows()
        print("Demo ended.")


if __name__ == "__main__":
    main()
