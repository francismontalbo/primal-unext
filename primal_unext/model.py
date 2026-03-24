"""Paper-aligned PRIMAL-UNeXt model builder."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as L
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    DepthwiseConv2D,
    Input,
    LayerNormalization,
    MaxPooling2D,
)

from primal_unext.layers import (
    AdaLNIN,
    ConfidenceGate,
    EdgeHead,
    EdgeRefine,
    GRNS,
    LGM,
    LKAplus,
    ResAdd,
    SCRoPEMHSA,
    SinCos2DPosEnc,
    StableSkipGate,
    SwiGLUFFN,
    TCSM,
)


def _tag(base: str, suffix: str) -> str:
    return f"{base}_{suffix}"


def aaspp(x, num_filters, rate_scale=1, name="AASPP"):
    b1 = Conv2D(num_filters, 3, padding="same", use_bias=False, dilation_rate=1 * rate_scale, name=_tag(f"{name}_conv_d1", "start"))(x)
    b1 = BatchNormalization(name=f"{name}_bn_d1")(b1)
    b1 = AdaLNIN(name=f"{name}_adln_d1")(b1)

    b6 = Conv2D(num_filters, 3, padding="same", use_bias=False, dilation_rate=6 * rate_scale, name=f"{name}_conv_d6")(x)
    b6 = BatchNormalization(name=f"{name}_bn_d6")(b6)
    b6 = AdaLNIN(name=f"{name}_adln_d6")(b6)

    b12 = Conv2D(num_filters, 3, padding="same", use_bias=False, dilation_rate=12 * rate_scale, name=f"{name}_conv_d12")(x)
    b12 = BatchNormalization(name=f"{name}_bn_d12")(b12)
    b12 = AdaLNIN(name=f"{name}_adln_d12")(b12)

    b18 = Conv2D(num_filters, 3, padding="same", use_bias=False, dilation_rate=18 * rate_scale, name=f"{name}_conv_d18")(x)
    b18 = BatchNormalization(name=f"{name}_bn_d18")(b18)
    b18 = AdaLNIN(name=f"{name}_adln_d18")(b18)

    y_stack = tf.stack([b1, b6, b12, b18], axis=-1)
    s = tf.reduce_mean(y_stack, axis=[1, 2, 3])
    g = L.Dense(max(4, num_filters // 8), activation="relu", name=f"{name}_mix_fc1")(s)
    w = L.Dense(4, activation="softmax", name=f"{name}_mix_fc2")(g)
    w = tf.reshape(w, (-1, 1, 1, 1, 4))
    y = tf.reduce_sum(y_stack * w, axis=-1)

    return Conv2D(num_filters, 1, padding="same", use_bias=False, name=_tag(f"{name}_proj", "end"))(y)


def primal_block(x, n_filter, strides=1, expansion_factor=3, name="E1_b0", mark_start=False, mark_end=False):
    sc = x
    dw_name = _tag(f"{name}_dw7", "start") if mark_start else f"{name}_dw7"
    add_name = _tag(f"{name}_add", "end") if mark_end else f"{name}_add"

    x = DepthwiseConv2D(7, strides=strides, padding="same", use_bias=False, name=dw_name)(x)
    x = AdaLNIN(name=f"{name}_adln")(x)
    x = Conv2D(int(n_filter * expansion_factor), 1, padding="same", use_bias=False, name=f"{name}_pw_expand")(x)
    x = Activation("gelu", name=f"{name}_gelu")(x)
    x = GRNS(name=f"{name}_grns")(x)
    x = Conv2D(n_filter, 1, padding="same", use_bias=False, name=f"{name}_pw_proj")(x)

    if (sc.shape[-1] != n_filter) or (strides != 1):
        sc = Conv2D(n_filter, 1, strides=strides, padding="same", use_bias=False, name=f"{name}_sc_proj")(sc)
        sc = AdaLNIN(name=f"{name}_sc_adln")(sc)

    return Add(name=add_name)([x, sc])


def stage_block(x, filters, blocks=2, stride_first=1, name="E1"):
    x = primal_block(x, n_filter=filters, strides=stride_first, expansion_factor=3, name=f"{name}_b0", mark_start=True, mark_end=(blocks == 1))
    for i in range(1, blocks):
        x = primal_block(x, n_filter=filters, strides=1, expansion_factor=3, name=f"{name}_b{i}", mark_end=(i == blocks - 1))
    return x


def scrope_block(tokens, h, w, num_heads=8, mlp_ratio=4.0, name="SCRoPE1"):
    c = int(tokens.shape[-1])
    key_dim = max(8, c // max(1, num_heads))
    key_dim += (4 - key_dim % 4) % 4

    y = LayerNormalization(name=_tag(f"{name}_ln1", "start"))(tokens)
    attn = SCRoPEMHSA(num_heads=num_heads, key_dim=key_dim, H=h, W=w, out_dim=c, name=f"{name}_mhsa")(y)
    tokens = ResAdd(name=f"{name}_ResAdd_mhsa")([tokens, attn])

    y = LayerNormalization(name=f"{name}_ln2")(tokens)
    ffn = SwiGLUFFN(expansion=mlp_ratio, name=f"{name}_ffn")(y)
    return ResAdd(name=_tag(f"{name}_ResAdd_ffn", "end"))([tokens, ffn])


def bridge_branch(x, tokens, dilation, heads, name="B1", tcsm_ratio=16):
    ch = int(x.shape[-1])

    x_pre = Conv2D(ch, 3, padding="same", use_bias=False, dilation_rate=dilation, name=_tag(f"{name}_preconv_d{dilation}", "start"))(x)
    x_aspp = aaspp(x_pre, ch, rate_scale=1, name=f"{name}_AASPP")

    x_lgm = LGM(num_memory=8, expand=4.0, num_heads=4, ffn_expand=2.0, use_tokens=True, name=f"{name}_LGM")([x_aspp, tokens])
    x = ResAdd(name=f"{name}_ResAdd_LGM")([x_aspp, x_lgm])

    x_tcsm = TCSM(ratio=tcsm_ratio, use_tokens=True, name=f"{name}_TCSM")([x, tokens])
    x = ResAdd(name=f"{name}_ResAdd_TCSM")([x, x_tcsm])

    x_lka = LKAplus(name=f"{name}_LKAplus")(x)
    x = ResAdd(name=f"{name}_ResAdd_LKAplus")([x, x_lka])

    hm = int(x.shape[1])
    wm = int(x.shape[2])

    t = L.Reshape((-1, ch), name=f"{name}_mapMHSA_flatten")(x)
    t = LayerNormalization(name=f"{name}_mapMHSA_ln")(t)

    key_dim = max(8, ch // max(1, heads))
    key_dim += (4 - key_dim % 4) % 4

    attn = SCRoPEMHSA(num_heads=heads, key_dim=key_dim, H=hm, W=wm, out_dim=ch, name=f"{name}_mapMHSA_attn")(t)
    attn = L.Reshape((hm, wm, ch), name=f"{name}_mapMHSA_unflatten")(attn)
    x = ResAdd(name=f"{name}_ResAdd_mapMHSA")([x, attn])

    x2 = AdaLNIN(name=f"{name}_mapFFN_adln")(x)
    ffn = SwiGLUFFN(expansion=2.0, name=f"{name}_mapFFN_ffn")(x2)
    return ResAdd(name=_tag(f"{name}_ResAdd_mapFFN", "end"))([x2, ffn])


def stem_block(x, n_filter=16):
    x = Conv2D(n_filter, 3, padding="same", use_bias=False, name=_tag("Stem_c1", "start"))(x)
    x = BatchNormalization(name="Stem_bn")(x)
    x = Activation("relu", name="Stem_relu")(x)
    return Conv2D(n_filter, 3, padding="same", use_bias=False, name=_tag("Stem_c2", "end"))(x)


def up2(x, filters, name):
    x = Conv2DTranspose(filters, 2, strides=2, padding="same", use_bias=False, name=_tag(f"{name}_tconv", "start"))(x)
    x = BatchNormalization(name=f"{name}_bn")(x)
    return Activation("relu", name=_tag(f"{name}_relu", "end"))(x)


def build_primal_unext(image_size=128, in_channels=3, num_classes=1, model_name="PRIMAL_UNeXt"):
    if image_size % 8 != 0:
        raise ValueError("image_size must be divisible by 8 (E2–E4 downsample by total factor 8).")

    inp = Input((image_size, image_size, in_channels), name="image")

    x0 = stem_block(inp, n_filter=16)

    e1 = stage_block(x0, 16, blocks=2, stride_first=1, name="E1")
    e2 = stage_block(e1, 16, blocks=2, stride_first=2, name="E2")
    e3 = stage_block(e2, 24, blocks=2, stride_first=2, name="E3")
    e4 = stage_block(e3, 32, blocks=2, stride_first=2, name="E4")

    grid = image_size // 8

    vp = Conv2D(128, 3, padding="same", use_bias=False, name="VP")(e4)
    vp = LayerNormalization(name="VP_LN")(vp)
    vp = SinCos2DPosEnc(name="VP_PosEnc")(vp)

    tok4 = L.Reshape((grid * grid, 128, 1), name="TOK")(vp)
    tok = L.Reshape((grid * grid, 128), name="TOK_seq")(tok4)

    tok = scrope_block(tok, h=grid, w=grid, num_heads=8, mlp_ratio=4.0, name="SCRoPE1")
    tok = scrope_block(tok, h=grid, w=grid, num_heads=8, mlp_ratio=4.0, name="SCRoPE2")

    tok_out = L.Reshape((grid * grid, 128, 1), name="TOK_out")(tok)
    tok_map = L.Reshape((grid, grid, 128), name="TOK_unflatten")(tok)

    va = Conv2D(32, 1, padding="same", use_bias=False, name="VA")(tok_map)

    b1 = bridge_branch(va, tokens=tok_out, dilation=3, heads=3, name="B1", tcsm_ratio=16)

    b2_in = MaxPooling2D(2, 2, name=_tag("B2_pool", "start"))(va)
    b2 = bridge_branch(b2_in, tokens=tok_out, dilation=6, heads=6, name="B2", tcsm_ratio=16)

    bu = Conv2DTranspose(32, 2, strides=2, padding="same", use_bias=False, name="BU")(b2)
    bc = Concatenate(name="BC")([b1, bu])

    d1u = up2(bc, 32, name="D1_Up")
    g1 = StableSkipGate(name="D1_SG_E3")([e3, d1u])
    d1 = Concatenate(name="D1_Cat")([d1u, g1])
    d1 = stage_block(d1, 32, blocks=1, stride_first=1, name="D1_Ref")

    d2u = up2(d1, 24, name="D2_Up")
    g2 = StableSkipGate(name="D2_SG_E2")([e2, d2u])
    d2 = Concatenate(name="D2_Cat")([d2u, g2])
    d2 = stage_block(d2, 24, blocks=1, stride_first=1, name="D2_Ref")

    d3u = up2(d2, 16, name="D3_Up")
    g3 = StableSkipGate(name="D3_SG_E1")([e1, d3u])
    d3 = Concatenate(name="D3_Cat")([d3u, g3])
    d3 = stage_block(d3, 16, blocks=1, stride_first=1, name="D3_Ref")

    edge = EdgeHead(name="H_E")(d3)
    conf = ConfidenceGate(name="H_C")(d3)

    logits = Conv2D(num_classes, 1, padding="same", use_bias=True, name=_tag("H_L", "start"))(d3)
    logits = EdgeRefine(num_classes=num_classes, alpha_init=0.05, name="H_R")([logits, edge, conf])

    activation = "sigmoid" if num_classes == 1 else "softmax"
    out = Activation(activation, name=_tag("H_R_sigma", "end"))(logits)

    return Model(inp, out, name=model_name)
