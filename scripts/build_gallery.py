#!/usr/bin/env python3
"""
Build Identity Gallery

Reads enrollment photographs from data/identities/, detects faces,
generates ArcFace embeddings, and saves them to models/face_gallery/.

Usage:
    python scripts/build_gallery.py

Directory structure expected:
    data/identities/
        shriram/
            01.jpg
            02.jpg
            ...
        person_02/
            01.jpg
            ...
"""
import sys
import os
import logging
from pathlib import Path


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
from tqdm import tqdm

from src.face.face_detector import FaceDetector
from src.face.recognizer import FaceRecognizer
from src.face.gallery import IdentityGallery

# ── Configuration ──────────────────────────────────────────────────────────

IDENTITIES_DIR = PROJECT_ROOT / "data" / "identities"
GALLERY_DIR = PROJECT_ROOT / "models" / "face_gallery"
SIMILARITY_THRESHOLD = 0.45  # Same threshold used during inference

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Logging Setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device_info() -> str:
    """Print device information."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return f"CUDA ({device_name})"
        else:
            return "CPU (no CUDA available)"
    except ImportError:
        return "CPU (PyTorch not installed)"


def discover_identities(identities_dir: Path) -> dict:
    """
    Discover all identity folders and their images.

    Returns:
        Dict mapping identity_name -> list of image paths.
    """
    identities = {}

    if not identities_dir.exists():
        logger.error(f"Identities directory not found: {identities_dir}")
        logger.info(f"Create it and add identity folders with enrollment photos.")
        return identities

    for identity_dir in sorted(identities_dir.iterdir()):
        if not identity_dir.is_dir():
            continue

        # Skip hidden directories
        if identity_dir.name.startswith("."):
            continue

        images = []
        for img_path in sorted(identity_dir.iterdir()):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(img_path)

        if images:
            identities[identity_dir.name] = images
        else:
            logger.warning(
                f"Identity '{identity_dir.name}' has no valid images. Skipping."
            )

    return identities


def main():
    """Main gallery building pipeline."""
    print("=" * 60)
    print("  L&T CCTV AI - Stage 1: Build Identity Gallery")
    print("=" * 60)
    print()

    # Device info
    device = get_device_info()
    print(f"Device: {device}")
    print()

    # Discover identities
    identities = discover_identities(IDENTITIES_DIR)

    if not identities:
        print(f"No identities found in {IDENTITIES_DIR}")
        print()
        print("Please add identity folders with enrollment photos:")
        print(f"  {IDENTITIES_DIR}/")
        print(f"      person_name/")
        print(f"          01.jpg")
        print(f"          02.jpg")
        print(f"          ...")
        print()
        sys.exit(1)

    total_images = sum(len(imgs) for imgs in identities.values())
    print(f"Found {len(identities)} identities with {total_images} total images:")
    for name, imgs in identities.items():
        print(f"  {name}: {len(imgs)} images")
    print()

    # Load models
    print("Loading face detection model (SCRFD)...")
    face_detector = FaceDetector()
    face_detector.load()

    print("Loading face recognition model (ArcFace)...")
    face_recognizer = FaceRecognizer()
    face_recognizer.load_from_shared(face_detector._analysis)
    print("Models loaded successfully.")
    print()

    # Build gallery
    gallery = IdentityGallery(
        similarity_threshold=SIMILARITY_THRESHOLD,
        gallery_dir=str(GALLERY_DIR),
    )

    all_embeddings = []
    all_identities = []
    skipped_count = 0

    for identity_name, image_paths in identities.items():
        print(f"Processing identity: {identity_name}")
        for img_path in tqdm(image_paths, desc=f"  {identity_name}", unit="img"):
            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"  Could not read: {img_path}")
                skipped_count += 1
                continue

            # Detect faces
            faces = face_detector.detect(frame)

            if len(faces) == 0:
                logger.warning(f"  No face detected in: {img_path.name}")
                skipped_count += 1
                continue

            # Use the face with highest confidence
            best_face = max(faces, key=lambda f: f.confidence)

            # Generate embedding
            embeddings = face_recognizer.generate_embeddings_batch(frame, [best_face])

            if len(embeddings) == 0:
                logger.warning(f"  Could not generate embedding for: {img_path.name}")
                skipped_count += 1
                continue

            all_embeddings.append(embeddings[0].embedding)
            all_identities.append(identity_name)

    print()

    if len(all_embeddings) == 0:
        logger.error("No valid embeddings were generated. Gallery not built.")
        sys.exit(1)

    # Build and save gallery
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    gallery.build_from_embeddings(embeddings_array, all_identities)
    gallery.save()

    # Summary
    print("=" * 60)
    print("  Gallery Built Successfully!")
    print("=" * 60)
    print()
    print(f"  Identities: {gallery.num_identities}")
    print(f"  Embeddings: {gallery.num_embeddings}")
    print(f"  Skipped:    {skipped_count}")
    print(f"  Gallery:    {GALLERY_DIR}")
    print(f"  Threshold:  {SIMILARITY_THRESHOLD}")
    print()
    print(f"  Files:")
    print(f"    {GALLERY_DIR / 'embeddings.npy'}")
    print(f"    {GALLERY_DIR / 'identities.json'}")
    print()


if __name__ == "__main__":
    main()
