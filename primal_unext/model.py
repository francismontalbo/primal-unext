"""PRIMAL-UNeXt model definition in module-style API.

This module keeps the same building intent used in the notebook implementation,
while exposing class-based blocks for easier maintenance and testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as L

from primal_unext.layers import AdaLNIN, GRNS, ResAdd


@dataclass(frozen=True)
class PRIMALUNeXtConfig:
    input_shape: tuple[int, int, int] = (128, 128, 3)
    num_classes: int = 1
    base_filters: int = 16
    name: str = "PRIMAL_UNeXt"


class ConvNormAct(L.Layer):
    def __init__(self, filters: int, kernel_size: int = 3, name: str = "conv_norm_act"):
        super().__init__(name=name)
        self.conv = L.Conv2D(filters, kernel_size, padding="same", use_bias=False, name=f"{name}_conv")
        self.bn = L.BatchNormalization(name=f"{name}_bn")
        self.act = L.Activation("swish", name=f"{name}_act")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)


class EncoderStage(L.Layer):
    def __init__(self, filters: int, stage_name: str):
        super().__init__(name=stage_name)
        self.pre = ConvNormAct(filters, 3, name=f"{stage_name}_pre")
        self.dw = L.DepthwiseConv2D(7, padding="same", use_bias=False, name=f"{stage_name}_dw7")
        self.norm = AdaLNIN(name=f"{stage_name}_adalin")
        self.post = ConvNormAct(filters, 3, name=f"{stage_name}_post")
        self.pool = L.MaxPooling2D(pool_size=2, name=f"{stage_name}_pool")

    def call(self, x: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.pre(x, training=training)
        x = self.dw(x)
        x = self.norm(x)
        x = self.post(x, training=training)
        skip = x
        x = self.pool(x)
        return x, skip


class AASPPBridge(L.Layer):
    """Bridge block aligned with notebook note: Conv(3x3) -> AdaLNIN branches."""

    def __init__(self, filters: int, name: str = "bridge"):
        super().__init__(name=name)
        self.branch_rates = (1, 2, 3, 5)
        self.branch_convs = [
            L.Conv2D(filters, 3, padding="same", dilation_rate=r, use_bias=False, name=f"{name}_b{i}_conv")
            for i, r in enumerate(self.branch_rates, start=1)
        ]
        self.branch_norms = [AdaLNIN(name=f"{name}_b{i}_adalin") for i in range(1, 5)]
        self.branch_acts = [L.Activation("swish", name=f"{name}_b{i}_act") for i in range(1, 5)]

        self.concat = L.Concatenate(name=f"{name}_concat")
        self.fuse_conv = L.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_fuse_conv")
        self.fuse_bn = L.BatchNormalization(name=f"{name}_fuse_bn")
        self.fuse_act = L.Activation("swish", name=f"{name}_fuse_act")

        self.res_path = ConvNormAct(filters, 3, name=f"{name}_res")
        self.res_grns = GRNS(name=f"{name}_grns")
        self.res_add = ResAdd(name=f"{name}_resadd")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        branches = []
        for conv, norm, act in zip(self.branch_convs, self.branch_norms, self.branch_acts):
            b = conv(x)
            b = norm(b)
            b = act(b)
            branches.append(b)

        x = self.concat(branches)
        x = self.fuse_conv(x)
        x = self.fuse_bn(x, training=training)
        x = self.fuse_act(x)

        residual = self.res_path(x, training=training)
        residual = self.res_grns(residual)
        return self.res_add((x, residual))


class DecoderStage(L.Layer):
    def __init__(self, filters: int, stage_name: str):
        super().__init__(name=stage_name)
        self.up = L.Conv2DTranspose(filters, kernel_size=2, strides=2, padding="same", name=f"{stage_name}_up")
        self.concat = L.Concatenate(name=f"{stage_name}_concat")
        self.block1 = ConvNormAct(filters, 3, name=f"{stage_name}_c1")
        self.block2 = ConvNormAct(filters, 3, name=f"{stage_name}_c2")

    def call(self, x: tf.Tensor, skip: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.up(x)
        x = self.concat([x, skip])
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        return x


class PRIMALUNeXt(Model):
    """Class-style PRIMAL-UNeXt model (Keras subclass API, PyTorch-like structure)."""

    def __init__(self, config: PRIMALUNeXtConfig):
        super().__init__(name=config.name)
        self.config = config
        b = config.base_filters

        self.stem = ConvNormAct(b, 3, name="stem")

        self.enc1 = EncoderStage(b, stage_name="enc1")
        self.enc2 = EncoderStage(b * 2, stage_name="enc2")
        self.enc3 = EncoderStage(b * 4, stage_name="enc3")
        self.enc4 = EncoderStage(b * 8, stage_name="enc4")

        self.bridge = AASPPBridge(b * 16, name="bridge")

        self.dec1 = DecoderStage(b * 8, stage_name="dec1")
        self.dec2 = DecoderStage(b * 4, stage_name="dec2")
        self.dec3 = DecoderStage(b * 2, stage_name="dec3")
        self.dec4 = DecoderStage(b, stage_name="dec4")

        activation = "sigmoid" if config.num_classes == 1 else "softmax"
        self.head = L.Conv2D(
            config.num_classes,
            kernel_size=1,
            padding="same",
            activation=activation,
            name="segmentation_head",
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.stem(x, training=training)

        x, s1 = self.enc1(x, training=training)
        x, s2 = self.enc2(x, training=training)
        x, s3 = self.enc3(x, training=training)
        x, s4 = self.enc4(x, training=training)

        x = self.bridge(x, training=training)

        x = self.dec1(x, s4, training=training)
        x = self.dec2(x, s3, training=training)
        x = self.dec3(x, s2, training=training)
        x = self.dec4(x, s1, training=training)

        return self.head(x)


def build_primal_unext(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 1,
    base_filters: int = 16,
    name: str = "PRIMAL_UNeXt",
) -> Model:
    """Factory that returns a built Keras model ready for compile/train."""
    cfg = PRIMALUNeXtConfig(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        name=name,
    )
    module = PRIMALUNeXt(cfg)
    inputs = L.Input(shape=input_shape, name="input")
    outputs = module(inputs)
    return Model(inputs=inputs, outputs=outputs, name=name)
