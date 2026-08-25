"""
Action Recognition Model (Stage 2)

Sets up a pretrained VideoMAE model fine-tuned for 13-class action recognition.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import VideoMAEForVideoClassification, VideoMAEConfig

from src.action.config import (
    ACTION_CLASSES,
    MODEL_NAME,
    MODEL_SAVE_DIR,
    NUM_ACTION_CLASSES,
    NUM_FRAMES,
)

logger = logging.getLogger(__name__)


def create_action_model(
    num_classes: int = NUM_ACTION_CLASSES,
    pretrained: bool = True,
    model_name: str = MODEL_NAME,
) -> VideoMAEForVideoClassification:
    """
    Create a VideoMAE model fine-tuned for action classification.

    Uses a pretrained VideoMAE backbone and replaces the classification head
    with one outputting `num_classes` logits.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load pretrained backbone weights.
        model_name: HuggingFace model identifier.

    Returns:
        VideoMAEForVideoClassification model.
    """
    logger.info(f"Creating VideoMAE model: {model_name} -> {num_classes} classes")

    if pretrained:
        # Load pretrained model and reinitialize the classifier head
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        logger.info("Loaded pretrained VideoMAE with new classification head.")
    else:
        # Build from config
        config = VideoMAEConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        model = VideoMAEForVideoClassification(config)
        logger.info("Created VideoMAE from config (no pretrained weights).")

    # Log model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    val_loss: float,
    class_to_idx: dict,
    config: dict,
    save_path: Path,
    is_best: bool = False,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        epoch: Current epoch number.
        val_accuracy: Current validation accuracy.
        val_loss: Current validation loss.
        class_to_idx: Class name to index mapping.
        config: Training configuration dict.
        save_path: Path to save the checkpoint.
        is_best: Whether this is the best checkpoint so far.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
        "class_to_idx": class_to_idx,
        "config": config,
        "action_classes": ACTION_CLASSES,
    }

    # Save as latest
    latest_path = save_path / "latest_checkpoint.pt"
    torch.save(checkpoint, latest_path)

    # Save as best
    if is_best:
        best_path = save_path / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model (val_acc={val_accuracy:.4f}) to {best_path}")

    # Always save latest
    logger.debug(f"Saved checkpoint (epoch {epoch}) to {latest_path}")


def load_checkpoint(
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> dict:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file. If None, loads best_model.pt.
        device: Device to load the checkpoint onto.

    Returns:
        Dictionary containing model state, config, class mappings, etc.
    """
    if checkpoint_path is None:
        checkpoint_path = MODEL_SAVE_DIR / "best_model.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

    return checkpoint


def load_trained_model(
    checkpoint_path: Optional[Path] = None,
    device: str = "cpu",
) -> tuple:
    """
    Load a fully trained model ready for inference.

    Args:
        checkpoint_path: Path to the checkpoint. If None, loads best.
        device: Target device.

    Returns:
        (model, class_to_idx, config) tuple.
    """
    checkpoint = load_checkpoint(checkpoint_path, device)

    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(class_to_idx)
    config = checkpoint.get("config", {})

    # Create model with same architecture
    model = create_action_model(num_classes=num_classes, pretrained=False)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(
        f"Model loaded: {num_classes} classes, "
        f"epoch={checkpoint.get('epoch', '?')}, "
        f"val_acc={checkpoint.get('val_accuracy', '?')}"
    )

    return model, class_to_idx, config
