#!/usr/bin/env python3
"""
L&T CCTV AI - Stage 2: Action Recognition Training

Trains a VideoMAE model for 13-class action recognition on the dataset
in archive/. Automatically discovers the dataset, builds train/val splits,
trains the model, evaluates it, and saves checkpoints.

Usage:
    python train_action.py
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure onnxruntime can find CUDA DLLs (needed if insightface is imported later)
def _setup_cuda_path():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.isdir(torch_lib) and torch_lib not in os.environ.get('PATH', ''):
            os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(torch_lib)
    except ImportError:
        pass
_setup_cuda_path()
del _setup_cuda_path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.action.config import (
    ACTION_CLASSES,
    BATCH_SIZE,
    CHECKPOINT_EVERY,
    GRAD_ACCUM_STEPS,
    LEARNING_RATE,
    MODEL_SAVE_DIR,
    NUM_EPOCHS,
    NUM_WORKERS,
    NUM_ACTION_CLASSES,
    PATIENCE,
    RESULTS_DIR,
    WEIGHT_DECAY,
    WARMUP_EPOCHS,
)
from src.action.dataset import (
    ActionVideoDataset,
    collate_fn,
    discover_dataset,
)
from src.action.model import (
    create_action_model,
    save_checkpoint,
)

# ── Logging Setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Determine the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Device: CUDA — GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        logger.warning("Device: CPU — No CUDA GPU available. Training will be slow.")
    return device


def get_lr_scheduler(optimizer, num_training_steps: int, num_warmup_steps: int):
    """Create a linear warmup + cosine decay learning rate scheduler."""
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch with AMP mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device.type == "cuda")
    optimizer.zero_grad()

    for batch_idx, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device)
        labels = labels.to(device)

        # Forward pass with AMP
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(pixel_values=videos)
            logits = outputs.logits
            loss = criterion(logits, labels)
            scaled_loss = loss / GRAD_ACCUM_STEPS

        # Backward pass
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Step optimizer every GRAD_ACCUM_STEPS
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            optimizer.zero_grad()

        # Stats
        total_loss += loss.item() * videos.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Log progress
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader):
            batch_acc = 100.0 * correct / max(total, 1)
            batch_loss = total_loss / max(total, 1)
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                f"loss={batch_loss:.4f}, acc={batch_acc:.1f}%, lr={lr:.6f}"
            )

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Validate the model.

    Returns:
        (val_loss, val_accuracy, all_labels, all_predictions)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    use_amp = (device.type == "cuda")

    for videos, labels in dataloader:
        videos = videos.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(pixel_values=videos)
            logits = outputs.logits
            loss = criterion(logits, labels)

        total_loss += loss.item() * videos.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    return avg_loss, accuracy, all_labels, all_preds


def save_evaluation_results(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    epoch: int,
    val_accuracy: float,
    results_dir: Path,
):
    """Save detailed evaluation results to files."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # Classification report
    report = classification_report(
        all_labels,
        all_preds,
        target_names=ACTION_CLASSES,
        digits=4,
        zero_division=0,
    )

    report_path = results_dir / f"classification_report_epoch{epoch}.txt"
    with open(report_path, "w") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Validation Accuracy: {val_accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_path = results_dir / f"confusion_matrix_epoch{epoch}.npy"
    np.save(str(cm_path), cm)

    # Human-readable confusion matrix
    cm_text_path = results_dir / f"confusion_matrix_epoch{epoch}.txt"
    with open(cm_text_path, "w") as f:
        f.write(f"Confusion Matrix (Epoch {epoch})\n")
        f.write(f"Rows = True, Columns = Predicted\n\n")

        # Header
        header = " ".join(f"{name[:6]:>7s}" for name in ACTION_CLASSES)
        f.write(f"{'':>12s} {header}\n")

        for i, row in enumerate(cm):
            row_str = " ".join(f"{val:7d}" for val in row)
            f.write(f"{ACTION_CLASSES[i]:>12s} {row_str}\n")

    # Also compute per-class metrics
    per_class_path = results_dir / f"per_class_metrics_epoch{epoch}.txt"
    with open(per_class_path, "w") as f:
        f.write(f"Per-Class Metrics (Epoch {epoch})\n\n")
        f.write(f"{'Class':>12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}\n")
        f.write("-" * 55 + "\n")

        prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
        rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

        for i, name in enumerate(ACTION_CLASSES):
            support = int(np.sum(all_labels == i))
            f.write(
                f"{name:>12s} {prec[i]:10.4f} {rec[i]:10.4f} {f1[i]:10.4f} {support:10d}\n"
            )

        f.write("-" * 55 + "\n")
        f.write(
            f"{'Macro Avg':>12s} "
            f"{np.mean(prec):10.4f} {np.mean(rec):10.4f} {np.mean(f1):10.4f} {len(all_labels):10d}\n"
        )

    logger.info(f"  Evaluation results saved to {results_dir}")


def main():
    """Main training entry point."""
    print()
    print("=" * 65)
    print("  L&T CCTV AI - Stage 2: Action Recognition Training")
    print("=" * 65)
    print()

    # ── Device ─────────────────────────────────────────────────────────────
    device = get_device()
    print()

    # ── Discover Dataset ───────────────────────────────────────────────────
    print("[1/5] Discovering dataset...")
    try:
        train_samples, val_samples = discover_dataset()
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        sys.exit(1)

    if not train_samples:
        logger.error("No training samples found. Check dataset structure.")
        sys.exit(1)

    if not val_samples:
        logger.error("No validation samples found. Check split files.")
        sys.exit(1)

    # Class distribution
    from collections import Counter
    train_counts = Counter(s["label"] for s in train_samples)
    val_counts = Counter(s["label"] for s in val_samples)
    print(f"  Training samples:   {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")
    print(f"  Action classes:     {NUM_ACTION_CLASSES}")
    print()

    # ── Create DataLoaders ─────────────────────────────────────────────────
    print("[2/5] Creating data loaders...")
    train_dataset = ActionVideoDataset(
        train_samples,
        is_training=True,
    )
    val_dataset = ActionVideoDataset(
        val_samples,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print()

    # ── Create Model ───────────────────────────────────────────────────────
    print("[3/5] Creating VideoMAE model...")
    model = create_action_model(
        num_classes=NUM_ACTION_CLASSES,
        pretrained=True,
    )
    model = model.to(device)
    print(f"  Model: {model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'VideoMAE'}")
    print(f"  Classes: {NUM_ACTION_CLASSES}")
    print()

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    print("[4/5] Setting up optimizer and scheduler...")

    # Separate parameter groups: higher LR for classifier head, lower for backbone
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "heads" in name or "head" in name:
            head_params.append((name, param))
        else:
            backbone_params.append((name, param))

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in backbone_params if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
                "lr": LEARNING_RATE * 0.1,  # Lower LR for backbone
            },
            {
                "params": [p for n, p in backbone_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": LEARNING_RATE * 0.1,
            },
            {
                "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
                "lr": LEARNING_RATE,
            },
            {
                "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": LEARNING_RATE,
            },
        ],
    )

    # Learning rate scheduler: linear warmup + cosine decay
    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS
    scheduler = get_lr_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"  Optimizer: AdamW")
    print(f"  LR: {LEARNING_RATE} (backbone: {LEARNING_RATE * 0.1})")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs")
    print(f"  Batch size: {BATCH_SIZE} (accum steps: {GRAD_ACCUM_STEPS}, effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print(f"  AMP Mixed Precision: Enabled ({device.type})")
    print()

    # ── Class-to-index mapping ─────────────────────────────────────────────
    class_to_idx = {name: idx for idx, name in enumerate(ACTION_CLASSES)}

    # Training config to save with checkpoints
    train_config = {
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_epochs": NUM_EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "label_smoothing": 0.1,
        "frame_size": 224,
        "num_frames": 16,
        "frame_stride": 2,
    }

    # ── Training Loop ──────────────────────────────────────────────────────
    print("[5/5] Starting training...")
    print()

    best_val_accuracy = 0.0
    patience_counter = 0
    training_log = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        logger.info(f"Epoch {epoch}/{NUM_EPOCHS}")
        logger.info("-" * 50)

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
        )

        # Validate
        val_loss, val_acc, val_labels, val_preds = validate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        # Log results
        logger.info(
            f"Epoch {epoch}/{NUM_EPOCHS} ({epoch_time:.1f}s): "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%"
        )

        training_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "time": epoch_time,
        })

        # Check if this is the best model
        is_best = val_acc > best_val_accuracy
        if is_best:
            best_val_accuracy = val_acc
            patience_counter = 0
            logger.info(f"  [*] New best model! Val accuracy: {val_acc:.2f}%")
        else:
            patience_counter += 1
            logger.info(
                f"  No improvement for {patience_counter}/{PATIENCE} epochs "
                f"(best: {best_val_accuracy:.2f}%)"
            )

        # Save checkpoint
        if epoch % CHECKPOINT_EVERY == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_accuracy=val_acc,
                val_loss=val_loss,
                class_to_idx=class_to_idx,
                config=train_config,
                save_path=MODEL_SAVE_DIR,
                is_best=is_best,
            )

        # Save evaluation results for best model
        if is_best:
            save_evaluation_results(
                val_labels, val_preds, epoch, val_acc, RESULTS_DIR
            )

        print(
            f"  Epoch {epoch:3d}/{NUM_EPOCHS}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:5.1f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:5.1f}% | "
            f"{'[*] NEW BEST' if is_best else ''}"
        )

        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    # ── Final Save ─────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Training Complete!")
    print("=" * 65)
    print()

    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        val_accuracy=val_acc,
        val_loss=val_loss,
        class_to_idx=class_to_idx,
        config=train_config,
        save_path=MODEL_SAVE_DIR,
        is_best=False,  # Don't overwrite best
    )

    # Save training log
    log_path = RESULTS_DIR / "training_log.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("L&T CCTV AI - Stage 2: Training Log\n")
        f.write("=" * 50 + "\n\n")
        for entry in training_log:
            f.write(
                f"Epoch {entry['epoch']:3d}: "
                f"train_loss={entry['train_loss']:.4f} "
                f"train_acc={entry['train_acc']:.1f}% "
                f"val_loss={entry['val_loss']:.4f} "
                f"val_acc={entry['val_acc']:.1f}% "
                f"time={entry['time']:.1f}s\n"
            )
        f.write(f"\nBest validation accuracy: {best_val_accuracy:.2f}%\n")

    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Checkpoints saved to:    {MODEL_SAVE_DIR}")
    print(f"Results saved to:        {RESULTS_DIR}")
    print(f"Training log:            {log_path}")
    print()
    print("Next steps:")
    print(f"  Test on a video:  python test_action.py --video path/to/video.mp4")
    print(f"  Run webcam demo:  python scripts/webcam_demo.py")
    print()


if __name__ == "__main__":
    main()
