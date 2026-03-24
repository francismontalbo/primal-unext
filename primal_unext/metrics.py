"""Metrics for binary and multiclass segmentation."""

from __future__ import annotations

import tensorflow as tf

EPS = tf.keras.backend.epsilon()


def _flatten_per_sample(x: tf.Tensor) -> tf.Tensor:
    return tf.reshape(x, [tf.shape(x)[0], -1])


def dice_binary(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = _flatten_per_sample(y_true)
    y_pred = _flatten_per_sample(y_pred)
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    denom = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


def iou_binary(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = _flatten_per_sample(y_true)
    y_pred = _flatten_per_sample(y_pred)
    inter = tf.reduce_sum(y_true * y_pred, axis=1)
    union = tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


def precision_binary(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = _flatten_per_sample(y_true)
    y_pred = _flatten_per_sample(y_pred)
    tp = tf.reduce_sum(y_true * y_pred, axis=1)
    pred_pos = tf.reduce_sum(y_pred, axis=1)
    return tf.reduce_mean((tp + EPS) / (pred_pos + EPS))
