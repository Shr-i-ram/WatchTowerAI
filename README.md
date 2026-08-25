# L&T CCTV AI — Integrated Monitoring Prototype

An AI-based employee monitoring proof of concept built for L&T's CCTV surveillance system.

**Stage 1**: Real-time person detection + face-based identity recognition  
**Stage 2**: Scene-level action recognition (13 actions → NORMAL/SUSPICIOUS classification)  
**Stage 3**: BoT-SORT person tracking, entry/exit monitoring & activity logging

---

## Table of Contents

1. [Architecture](#architecture)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Stage 1: Person Identification](#stage-1-person-identification)
6. [Stage 2: Action Recognition](#stage-2-action-recognition)
7. [Stage 3: Person Tracking & Monitoring](#stage-3-person-tracking--monitoring)
8. [Configuration](#configuration)
9. [Pipeline Explained](#pipeline-explained)
10. [Monitoring Logs](#monitoring-logs)
11. [Known Limitations](#known-limitations)
12. [Project Structure](#project-structure)
13. [Roadmap](#roadmap)

---

## Architecture

```
                         LIVE WEBCAM
                              │
                              ▼
                    ┌─────────────────┐
                    │   YOLO11m       │  Person detection
                    │   + BoT-SORT    │  + Persistent tracking
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────────┐
              │              │                  │
              ▼              ▼                  ▼
      STAGE 1: IDENTITY   STAGE 2: ACTION    STAGE 3: SESSION
              │                   │                  │
     YOLO → Face Detection  Rolling Buffer     SessionManager
            → ArcFace             │                  │
            → Gallery             ▼                  ▼
              │            VideoMAE Model      Entry / Exit
              ▼                   │            Identity Lock
       Known / Unknown           ▼            Activity Timeline
                            Scene Action            │
                               │                    ▼
                               ▼            Monitoring Logger
                         NORMAL / SUSPICIOUS  (structured JSON)
```

### Stage 3 — Session Lifecycle

```
Person enters camera view
      ↓
New Track ID (BoT-SORT)
      ↓
Entry confirmed (3 consecutive frames)
      ↓
PersonSession created
      ↓
  ┌──────────────────────────────────────┐
  │ While person is visible:             │
  │   • Update identity (face recog.)    │
  │   • Record scene actions (Stage 2)   │
  │   • Track ID persists across frames  │
  └──────────────────────────────────────┘
      ↓
Person leaves camera view
      ↓
Grace period (30 frames ≈ 1s)
      ↓
  ├── Person returns → Continue session
  │
  └── Timeout expires → EXIT recorded
                         ↓
                   Session finalized
                         ↓
                   Saved to JSON log
```

---

## Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4060 Ti / RTX 4060 Laptop)
- **Webcam**: USB or built-in webcam (for live demo)
- **RAM**: ≥ 8 GB recommended
- **VRAM**: ≥ 4 GB for training, ≥ 2 GB for inference

### Software

- Python 3.13+ (tested on 3.13.2)
- CUDA 13.0 (via PyTorch) + cuDNN
- NVIDIA GPU drivers (latest recommended)
- Windows (primary), Linux (compatible)

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd lnt-cctv-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/macOS
```

### 3. Install PyTorch with CUDA 13.0

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify GPU availability

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Quick Start

```bash
# 1. Set up environment (see Installation above)

# 2. (Optional) Add enrollment photos for person identification
#    Place photos in data/identities/<name>/ and run:
python scripts/build_gallery.py

# 3. Train the action recognition model
python train_action.py

# 4. Run the complete integrated demo
python scripts/webcam_demo.py
```

---

## Stage 1: Person Identification

### Identity Enrollment

```bash
mkdir -p data/identities/shriram
cp photos/shriram_01.jpg data/identities/shriram/01.jpg
cp photos/shriram_02.jpg data/identities/shriram/02.jpg
```

### Building the Gallery

```bash
python scripts/build_gallery.py
```

---

## Stage 2: Action Recognition

### Action Classes

| # | Action | Status |
|---|--------|--------|
| 1 | Fall | SUSPICIOUS |
| 2 | Grab | SUSPICIOUS |
| 3 | Gun | SUSPICIOUS |
| 4 | Hit | SUSPICIOUS |
| 5 | Kick | SUSPICIOUS |
| 6 | LyingDown | SUSPICIOUS |
| 7 | Run | SUSPICIOUS |
| 8 | Sit | NORMAL |
| 9 | Stand | NORMAL |
| 10 | Sneak | NORMAL |
| 11 | Struggle | SUSPICIOUS |
| 12 | Throw | SUSPICIOUS |
| 13 | Walk | NORMAL |

### Training

```bash
python train_action.py
```

### Testing on a Video

```bash
python test_action.py --video path/to/video.mp4
```

---

## Stage 3: Person Tracking & Monitoring

### How It Works

Stage 3 extends the system with **persistent person tracking** and **session-based monitoring**:

1. **BoT-SORT Tracking**: The same YOLO detections are fed into BoT-SORT, which assigns persistent track IDs to each person. The same person keeps the same ID across consecutive frames.

2. **Session Lifecycle**: When a tracked person is confirmed (appears for 3+ consecutive frames), a `PersonSession` is created recording their entry time. When they leave and don't return within 30 frames (~1 second), the session is finalized with an exit time.

3. **Identity Stability**: Once a person is identified with high confidence (similarity ≥ 0.50), their identity is "locked in" and will not revert to UNKNOWN on a single missed face detection. This handles cases where a person turns away briefly.

4. **Activity Timeline**: While a person is present, scene-level actions from Stage 2 are recorded in their activity timeline. Consecutive frames with the same action are merged into single segments. These are explicitly labeled as **scene activities observed during the person's presence** — not actions performed by that person.

5. **Structured Logging**: Completed sessions are saved to JSON log files in the `logs/` directory. Each application run creates a new timestamped log file so sessions are never overwritten.

### Running the Complete System

```bash
python scripts/webcam_demo.py
```

The display shows:

```
┌──────────────────────────────────────────────┐
│  FPS: 28.4  |  Persons: 2                   │
│                                              │
│  ┌────────────────┐                          │
│  │  Shriram       │                          │
│  │  Track #7      │  ← person bounding box   │
│  └────────────────┘    with identity + ID    │
│                                              │
│  SCENE ACTION: WALK                          │
│  Confidence: 91.4%                           │
│  STATUS: NORMAL                              │
│  Active Tracks: 1                            │
└──────────────────────────────────────────────┘
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit (saves all monitoring logs) |
| `r` | Reload identity gallery |

### Monitoring Log Structure

Each completed session is saved with the following information:

```json
{
  "identity": "Shriram",
  "track_id": 7,
  "entry_time": "2026-08-25T10:31:04+00:00",
  "exit_time": "2026-08-25T10:33:10+00:00",
  "duration_seconds": 126.0,
  "scene_activity_timeline": [
    {
      "action": "WALK",
      "start_time": "2026-08-25T10:31:04+00:00",
      "end_time": "2026-08-25T10:31:20+00:00",
      "status": "NORMAL",
      "duration_seconds": 16.0
    },
    {
      "action": "RUN",
      "start_time": "2026-08-25T10:32:45+00:00",
      "end_time": "2026-08-25T10:32:52+00:00",
      "status": "SUSPICIOUS",
      "duration_seconds": 7.0
    }
  ],
  "scene_actions_observed": ["WALK", "RUN"],
  "suspicious_actions_observed": ["RUN"]
}
```

---

## Configuration

### Stage 3 Configuration (in `src/monitoring/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRACKER_CONFIG` | `"botsort.yaml"` | Ultralytics tracker config |
| `EXIT_GRACE_FRAMES` | `30` | Frames before exit is confirmed (~1s at 30fps) |
| `ENTRY_CONFIRM_FRAMES` | `3` | Frames before a track becomes a session |
| `IDENTITY_LOCK_THRESHOLD` | `0.50` | Similarity to lock identity (won't revert to UNKNOWN) |
| `IDENTITY_UNLOCK_FRAMES` | `60` | Frames without face before identity is re-evaluated |
| `ACTIVITY_MIN_SEGMENT_SECONDS` | `1.0` | Min duration for an activity segment |
| `LOG_DIR` | `logs/` | Directory for monitoring log files |

---

## Pipeline Explained

### Complete Three-Stage Pipeline

1. **YOLO11m** detects all persons in the frame
2. **BoT-SORT** assigns persistent track IDs across frames
3. For each tracked person, **SCRFD** detects their face within their bounding box
4. **ArcFace** generates a 512-d embedding for face recognition
5. The embedding is compared against the **identity gallery** using cosine similarity
6. **SessionManager** creates/updates/finalizes person sessions
7. **Stage 2 VideoMAE** classifies the overall scene action from a rolling frame buffer
8. Scene actions are recorded in each active person's **activity timeline**
9. On exit, sessions are saved to **structured JSON logs**

---

## Monitoring Logs

Logs are saved to `logs/session_YYYYMMDD_HHMMSS.json`.

Each application run creates a new file. Multiple runs produce multiple log files that can be inspected independently.

---

## Known Limitations

### Stage 1
- Identification only works when the face is visible
- Lighting and scale sensitivity
- Gallery rebuild replaces all embeddings

### Stage 2
- Scene-level only — actions are NOT attributed to individual people
- 16-frame temporal context
- Training dataset bias

### Stage 3
- Single camera only — no cross-camera ReID
- Temporary tracker ID switches may occur with heavy occlusion
- Scene activities recorded during a person's presence do NOT imply the person performed those actions

---

## Project Structure

```
lnt-cctv-ai/
├── archive/                          # Kaggle action recognition dataset
├── data/identities/                  # Stage 1: Enrollment photographs
├── logs/                             # Stage 3: Monitoring session logs
├── models/
│   ├── face_gallery/                 # Stage 1: Generated gallery
│   └── action_model/                 # Stage 2: Trained action model
├── results/                          # Stage 2: Training evaluation results
├── src/
│   ├── detection/person_detector.py  # YOLO11m person detection
│   ├── face/                         # SCRFD + ArcFace + Gallery
│   ├── action/                       # Stage 2: Action recognition
│   ├── tracking/tracker.py           # Stage 3: BoT-SORT tracker wrapper
│   ├── monitoring/                   # Stage 3: Session + Activity + Logger
│   │   ├── config.py                 # Stage 3 configuration
│   │   ├── session.py                # Person session lifecycle
│   │   ├── activity_timeline.py      # Action timeline per person
│   │   └── logger.py                 # Structured JSON log storage
│   ├── camera/webcam.py              # OpenCV webcam capture
│   └── visualization/overlay.py      # Bounding boxes + labels
├── scripts/
│   ├── build_gallery.py              # Stage 1: Build identity gallery
│   └── webcam_demo.py                # Integrated demo (Stage 1+2+3)
├── train_action.py                   # Stage 2: Train action recognition
├── test_action.py                    # Stage 2: Offline action test
└── requirements.txt
```

---

## Roadmap

### Stage 1 (Complete) ✅
- Real-time person detection (YOLO11m)
- Face detection and alignment (SCRFD)
- ArcFace identity recognition
- Identity gallery management
- Live webcam overlay with temporal smoothing

### Stage 2 (Complete) ✅
- Scene-level action recognition (VideoMAE)
- 13 action classes
- Configurable suspicious/normal classification
- Automated training pipeline
- Offline testing script

### Stage 3 (Complete) ✅
- BoT-SORT persistent person tracking
- Entry/exit detection with grace periods
- Identity stability and lock-in
- Scene activity timeline per person
- Structured JSON monitoring logs
- Graceful shutdown with session finalization

### Stage 4 (Planned)
- Person ReID across camera views
- Multi-camera identity association
- Per-person action recognition
- Employee database integration

### Stage 5 (Future)
- Alert and notification system
- Web dashboard for monitoring
- Cloud deployment and scaling
