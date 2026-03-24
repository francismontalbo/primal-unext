"""Utility helpers for reproducible experiments and model checks."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ExperimentConfig:
    image_height: int = 128
    image_width: int = 128
    channels: int = 3
    num_classes: int = 1
    batch_size: int = 4

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.image_height, self.image_width, self.channels)


def set_global_seed(seed: int = 42) -> None:
    """Set seeds across Python, NumPy, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_model_stats(model: tf.keras.Model) -> dict[str, float]:
    """Return parameter counts and approximate size in MB."""
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    non_trainable = int(np.sum([np.prod(v.shape) for v in model.non_trainable_weights]))
    total = trainable + non_trainable
    approx_mb = (total * 4) / (1024**2)
    return {
        "params_total": float(total),
        "params_trainable": float(trainable),
        "params_non_trainable": float(non_trainable),
        "approx_size_mb_fp32": float(approx_mb),
    }


def synthetic_batch(config: ExperimentConfig, batch_size: int | None = None):
    """Generate synthetic tensors for smoke testing."""
    b = batch_size or config.batch_size
    x = tf.random.uniform((b, *config.input_shape), minval=0.0, maxval=1.0, dtype=tf.float32)
    if config.num_classes == 1:
        y = tf.cast(tf.random.uniform((b, config.image_height, config.image_width, 1)) > 0.5, tf.float32)
    else:
        y_idx = tf.random.uniform(
            (b, config.image_height, config.image_width), minval=0, maxval=config.num_classes, dtype=tf.int32
        )
        y = tf.one_hot(y_idx, depth=config.num_classes, dtype=tf.float32)
    return x, y
