"""Production-friendly PRIMAL-UNeXt model builder."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.models import Model

from primal_unext.layers import AdaLNIN, GRNS, ResAdd


def conv_bn_act(x: tf.Tensor, filters: int, name: str, kernel_size: int = 3) -> tf.Tensor:
    x = L.Conv2D(filters, kernel_size, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = L.BatchNormalization(name=f"{name}_bn")(x)
    x = L.Activation("swish", name=f"{name}_act")(x)
    return x


def encoder_block(x: tf.Tensor, filters: int, name: str) -> tuple[tf.Tensor, tf.Tensor]:
    x = conv_bn_act(x, filters, name=f"{name}_c1")
    x = L.DepthwiseConv2D(7, padding="same", use_bias=False, name=f"{name}_dw7")(x)
    x = AdaLNIN(name=f"{name}_adalin")(x)
    x = conv_bn_act(x, filters, name=f"{name}_c2")
    skip = x
    x = L.MaxPooling2D(pool_size=2, name=f"{name}_pool")(x)
    return x, skip


def bottleneck_block(x: tf.Tensor, filters: int, name: str = "bridge") -> tf.Tensor:
    branches = []
    for i, rate in enumerate((1, 2, 3, 5), start=1):
        b = L.Conv2D(filters, 3, padding="same", dilation_rate=rate, use_bias=False, name=f"{name}_b{i}_conv")(x)
        b = AdaLNIN(name=f"{name}_b{i}_adalin")(b)
        b = L.Activation("swish", name=f"{name}_b{i}_act")(b)
        branches.append(b)

    x = L.Concatenate(name=f"{name}_concat")(branches)
    x = L.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_fuse_conv")(x)
    x = L.BatchNormalization(name=f"{name}_fuse_bn")(x)
    x = L.Activation("swish", name=f"{name}_fuse_act")(x)

    residual = conv_bn_act(x, filters, name=f"{name}_res")
    residual = GRNS(name=f"{name}_grns")(residual)
    x = ResAdd(name=f"{name}_resadd")((x, residual))
    return x


def decoder_block(x: tf.Tensor, skip: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = L.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same", name=f"{name}_up")(x)
    x = L.Concatenate(name=f"{name}_concat")([x, skip])
    x = conv_bn_act(x, filters, name=f"{name}_c1")
    x = conv_bn_act(x, filters, name=f"{name}_c2")
    return x


def build_primal_unext(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 1,
    base_filters: int = 32,
    name: str = "PRIMAL_UNeXt",
) -> Model:
    """Build PRIMAL-UNeXt for binary or multi-class segmentation."""
    inputs = L.Input(shape=input_shape, name="input")

    x = conv_bn_act(inputs, base_filters, name="stem")

    x, s1 = encoder_block(x, base_filters, name="enc1")
    x, s2 = encoder_block(x, base_filters * 2, name="enc2")
    x, s3 = encoder_block(x, base_filters * 4, name="enc3")

    x = bottleneck_block(x, base_filters * 8, name="bridge")

    x = decoder_block(x, s3, base_filters * 4, name="dec1")
    x = decoder_block(x, s2, base_filters * 2, name="dec2")
    x = decoder_block(x, s1, base_filters, name="dec3")

    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = L.Conv2D(num_classes, kernel_size=1, padding="same", activation=activation, name="segmentation_head")(x)

    return Model(inputs=inputs, outputs=outputs, name=name)
