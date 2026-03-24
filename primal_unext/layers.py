"""Custom layers used by PRIMAL-UNeXt."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L


def register_serializable(name: str | None = None):
    """Register class for model serialization when available."""

    def _decorator(cls):
        try:
            return tf.keras.utils.register_keras_serializable(
                package="primal_unext", name=name or cls.__name__
            )(cls)
        except Exception:
            return cls

    return _decorator


@register_serializable()
class GRNS(L.Layer):
    """Per-channel spatial standardization over height and width."""

    def __init__(self, epsilon: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) / tf.sqrt(var + self.epsilon)

    def get_config(self):
        return {"epsilon": self.epsilon, **super().get_config()}


@register_serializable()
class AdaLNIN(L.Layer):
    """Adaptive blend between instance norm and layer norm with learnable gate."""

    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(8, channels // 4)

        self.gamma = self.add_weight(
            name="gamma", shape=(channels,), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            name="beta", shape=(channels,), initializer="zeros", trainable=True
        )

        self.gap = L.GlobalAveragePooling2D(name=f"{self.name}_gap")
        self.gate_fc1 = L.Dense(hidden, activation="relu", name=f"{self.name}_gate_fc1")
        self.gate_fc2 = L.Dense(channels, activation="sigmoid", name=f"{self.name}_gate_fc2")
        super().build(input_shape)

    def _instance_norm(self, x: tf.Tensor) -> tf.Tensor:
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) / tf.sqrt(var + self.epsilon)

    def _layer_norm(self, x: tf.Tensor) -> tf.Tensor:
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - mean) / tf.sqrt(var + self.epsilon)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        gate = self.gate_fc2(self.gate_fc1(self.gap(x)))
        gate = tf.reshape(gate, (-1, 1, 1, tf.shape(x)[-1]))
        blended = gate * self._instance_norm(x) + (1.0 - gate) * self._layer_norm(x)
        return blended * self.gamma + self.beta

    def get_config(self):
        return {"epsilon": self.epsilon, **super().get_config()}


@register_serializable()
class ResAdd(L.Layer):
    """Residual add with learnable layer scale and gate scale."""

    def __init__(self, init_scale: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.init_scale = float(init_scale)

    def build(self, input_shape):
        channels = int(input_shape[0][-1])
        self.layer_scale = self.add_weight(
            name="layer_scale",
            shape=(channels,),
            initializer=tf.keras.initializers.Constant(self.init_scale),
            trainable=True,
        )
        self.gate = self.add_weight(
            name="gate",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        shortcut, residual = inputs
        scaled = residual * self.layer_scale
        gated = tf.sigmoid(self.gate) * scaled
        return shortcut + gated

    def get_config(self):
        return {"init_scale": self.init_scale, **super().get_config()}
