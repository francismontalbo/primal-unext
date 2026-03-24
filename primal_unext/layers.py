"""Custom layers and attention modules for PRIMAL-UNeXt."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
from tensorflow.keras.layers import Conv2D, Dense, DepthwiseConv2D, GlobalAveragePooling2D, LayerNormalization


def register_serializable(name: str | None = None):
    def deco(cls):
        return tf.keras.utils.register_keras_serializable(
            package="primal_unext", name=name or cls.__name__
        )(cls)

    return deco


def _rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return tf.concat([-x2, x1], axis=-1)


def _sincos_1d(length, dim):
    half = dim // 2
    i = tf.range(half, dtype=tf.float32)
    inv = 1.0 / (10000.0 ** (i / tf.cast(half, tf.float32)))
    pos = tf.cast(tf.range(length), tf.float32)[:, None] * inv[None, :]
    sin = tf.repeat(tf.sin(pos), repeats=2, axis=-1)
    cos = tf.repeat(tf.cos(pos), repeats=2, axis=-1)
    return sin, cos


@register_serializable()
class GRNS(L.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, x):
        m, v = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - m) / tf.sqrt(v + self.epsilon)

    def get_config(self):
        return {"epsilon": self.epsilon, **super().get_config()}


@register_serializable()
class AdaLNIN(L.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(c,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(c,), initializer="zeros", trainable=True)

        hidden = max(8, c // 4)
        self.gap = GlobalAveragePooling2D(name=f"{self.name}_gap")
        self.g_fc1 = Dense(hidden, activation="relu", name=f"{self.name}_g_fc1")
        self.g_fc2 = Dense(c, activation="sigmoid", name=f"{self.name}_g_fc2")
        super().build(input_shape)

    def _in(self, x):
        m, v = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - m) / tf.sqrt(v + self.epsilon)

    def _ln(self, x):
        m, v = tf.nn.moments(x, axes=[-1], keepdims=True)
        return (x - m) / tf.sqrt(v + self.epsilon)

    def call(self, x):
        s = self.gap(x)
        g = self.g_fc2(self.g_fc1(s))
        g = tf.reshape(g, (-1, 1, 1, tf.shape(x)[-1]))
        y = g * self._in(x) + (1.0 - g) * self._ln(x)
        return y * self.gamma + self.beta

    def get_config(self):
        return {"epsilon": self.epsilon, **super().get_config()}


@register_serializable()
class LayerScale(L.Layer):
    def __init__(self, init_value=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.init_value = float(init_value)

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(c,),
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return x * self.gamma

    def get_config(self):
        return {"init_value": self.init_value, **super().get_config()}


@register_serializable()
class GateScale(L.Layer):
    def __init__(self, init_logit=-5.0, temperature=1.0, l1=0.0, **kwargs):
        super().__init__(**kwargs)
        self.init_logit = float(init_logit)
        self.temperature = float(temperature)
        self.l1 = float(l1)

    def build(self, input_shape):
        self.logit = self.add_weight(
            name="logit",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init_logit),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=None):
        g = tf.sigmoid(self.logit / self.temperature)
        if self.l1 > 0:
            self.add_loss(self.l1 * tf.abs(g))
        return x * g

    def get_config(self):
        return {
            "init_logit": self.init_logit,
            "temperature": self.temperature,
            "l1": self.l1,
            **super().get_config(),
        }


@register_serializable()
class ResAdd(L.Layer):
    def __init__(self, ls_init=1e-4, gate_logit=-5.0, **kwargs):
        super().__init__(**kwargs)
        self.ls_init = float(ls_init)
        self.gate_logit = float(gate_logit)

    def build(self, input_shapes):
        self.ls = LayerScale(self.ls_init, name=f"{self.name}_ls")
        self.gate = GateScale(self.gate_logit, name=f"{self.name}_gate")
        self.add = L.Add(name=f"{self.name}_add")
        super().build(input_shapes)

    def call(self, inputs):
        base, branch = inputs
        branch = self.gate(self.ls(branch))
        return self.add([base, branch])

    def get_config(self):
        return {"ls_init": self.ls_init, "gate_logit": self.gate_logit, **super().get_config()}


@register_serializable(name="SCRoPE_MHSA")
class SCRoPEMHSA(L.Layer):
    def __init__(self, num_heads, key_dim, H=None, W=None, out_dim=None, eps=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.h = int(num_heads)
        kd = int(key_dim)
        kd += (4 - kd % 4) % 4
        self.d = kd
        self.H = None if H in (None, 0) else int(H)
        self.W = None if W in (None, 0) else int(W)
        self.out_dim = out_dim
        self.eps = float(eps)

    def build(self, input_shape):
        c = int(input_shape[-1])
        dall = self.h * self.d
        self.wq = Dense(dall, use_bias=False, name=f"{self.name}_wq")
        self.wk = Dense(dall, use_bias=False, name=f"{self.name}_wk")
        self.wv = Dense(dall, use_bias=False, name=f"{self.name}_wv")
        self.wo = Dense(self.out_dim or c, use_bias=False, name=f"{self.name}_wo")

        self.gq = self.add_weight(name="gq", shape=(self.h,), initializer="ones", trainable=True)
        self.gk = self.add_weight(name="gk", shape=(self.h,), initializer="ones", trainable=True)

        eye = np.eye(self.h, dtype="float32")
        self.w_pre = self.add_weight(
            name="w_pre",
            shape=(self.h, self.h),
            initializer=tf.keras.initializers.Constant(eye),
            trainable=True,
        )
        self.w_post = self.add_weight(
            name="w_post",
            shape=(self.h, self.h),
            initializer=tf.keras.initializers.Constant(eye),
            trainable=True,
        )

        self.tau = self.add_weight(
            name="tau",
            shape=(self.h,),
            initializer=tf.keras.initializers.Constant(1.0 / (self.d**0.5)),
            trainable=True,
        )
        self.hscale = self.add_weight(name="hscale", shape=(self.h,), initializer="ones", trainable=True)
        super().build(input_shape)

    def _rope2d(self, q, k, h, w):
        b = tf.shape(q)[0]
        t = tf.shape(q)[2]
        q = tf.reshape(q, [b, self.h, h, w, self.d])
        k = tf.reshape(k, [b, self.h, h, w, self.d])

        dy = self.d // 2
        qy, qx = tf.split(q, [dy, self.d - dy], axis=-1)
        ky, kx = tf.split(k, [dy, self.d - dy], axis=-1)

        def _apply(x, length, along="y"):
            d = tf.shape(x)[-1]
            pad = tf.math.floormod(d, 2)
            x = tf.cond(tf.equal(pad, 1), lambda: tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]]), lambda: x)
            de = tf.shape(x)[-1]
            sin, cos = _sincos_1d(length, tf.cast(de, tf.int32))

            if along == "y":
                sin = tf.reshape(sin, [1, 1, length, 1, de])
                cos = tf.reshape(cos, [1, 1, length, 1, de])
            else:
                sin = tf.reshape(sin, [1, 1, 1, length, de])
                cos = tf.reshape(cos, [1, 1, 1, length, de])

            out = x * cos + _rotate_half(x) * sin
            return out[..., :d]

        qy = _apply(qy, h, "y")
        ky = _apply(ky, h, "y")
        qx = _apply(qx, w, "x")
        kx = _apply(kx, w, "x")

        q = tf.reshape(tf.concat([qy, qx], axis=-1), [b, self.h, t, self.d])
        k = tf.reshape(tf.concat([ky, kx], axis=-1), [b, self.h, t, self.d])
        return q, k

    def call(self, x):
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]

        if (self.H is None) or (self.W is None):
            s = tf.cast(tf.sqrt(tf.cast(t, tf.float32)), tf.int32)
            h, w = s, s
        else:
            h, w = tf.constant(self.H, tf.int32), tf.constant(self.W, tf.int32)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        def _rs(y):
            y = tf.reshape(y, [b, -1, self.h, self.d])
            return tf.transpose(y, [0, 2, 1, 3])

        q = _rs(q)
        k = _rs(k)
        v = _rs(v)
        q, k = self._rope2d(q, k, h, w)

        def _l2n(y, g):
            y = y / (tf.norm(y, axis=-1, keepdims=True) + self.eps)
            return y * tf.reshape(g, [1, self.h, 1, 1])

        q = _l2n(q, self.gq)
        k = _l2n(k, self.gk)

        logits = tf.einsum("bhid,bhjd->bhij", q, k) * tf.reshape(self.tau, [1, self.h, 1, 1])
        logits = tf.einsum("bhij,hH->bHij", logits, self.w_pre)
        a = tf.nn.softmax(logits, axis=-1)
        a = tf.einsum("bhij,hH->bHij", a, self.w_post)
        y = tf.einsum("bhij,bhjd->bhid", a, v) * tf.reshape(self.hscale, [1, self.h, 1, 1])
        y = tf.reshape(tf.transpose(y, [0, 2, 1, 3]), [b, t, self.h * self.d])
        return self.wo(y)


@register_serializable()
class SwiGLUFFN(L.Layer):
    def __init__(self, expansion=4.0, **kwargs):
        super().__init__(**kwargs)
        self.expansion = float(expansion)

    def build(self, input_shape):
        ch = int(input_shape[-1])
        hid = int(self.expansion * ch)
        self.fc1 = Dense(hid * 2, use_bias=False, name=f"{self.name}_fc1")
        self.fc2 = Dense(ch, use_bias=False, name=f"{self.name}_fc2")
        super().build(input_shape)

    def call(self, x):
        u, v = tf.split(self.fc1(x), 2, axis=-1)
        return self.fc2(tf.nn.silu(u) * v)


@register_serializable(name="LGM")
class LGM(L.Layer):
    def __init__(self, num_memory=8, expand=4.0, num_heads=4, ffn_expand=2.0, use_tokens=True, **kwargs):
        super().__init__(**kwargs)
        self.num_memory = int(num_memory)
        self.expand = float(expand)
        self.num_heads = int(num_heads)
        self.ffn_expand = float(ffn_expand)
        self.use_tokens = bool(use_tokens)

    def build(self, input_shape):
        x_shape = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        c_in = int(x_shape[-1])
        self.c_out = c_in
        self.d = max(8, int(self.expand * c_in))

        self.proj_in = Dense(self.d, use_bias=False, name=f"{self.name}_proj_in")
        self.norm_tok1 = LayerNormalization(name=f"{self.name}_nt1")
        self.norm_tok2 = LayerNormalization(name=f"{self.name}_nt2")
        self.norm_mem1 = LayerNormalization(name=f"{self.name}_nm1")
        self.norm_mem2 = LayerNormalization(name=f"{self.name}_nm2")
        self.ln_ff = LayerNormalization(name=f"{self.name}_ln_ff")

        self.mem = self.add_weight(
            name="mem_slots", shape=(self.num_memory, self.d), initializer="glorot_uniform", trainable=True
        )

        key_dim = max(8, self.d // self.num_heads)
        self.q_proj = Dense(self.num_heads * key_dim, use_bias=False, name=f"{self.name}_q")
        self.k_proj = Dense(self.num_heads * key_dim, use_bias=False, name=f"{self.name}_k")
        self.v_proj = Dense(self.num_heads * key_dim, use_bias=False, name=f"{self.name}_v")
        self.o_proj = Dense(self.d, use_bias=False, name=f"{self.name}_o")

        self.ffn = SwiGLUFFN(expansion=self.ffn_expand, name=f"{self.name}_ffn")
        self.ls_r1 = LayerScale(1e-4, name=f"{self.name}_ls_r1")
        self.ls_w1 = LayerScale(1e-4, name=f"{self.name}_ls_w1")
        self.ls_ff = LayerScale(1e-4, name=f"{self.name}_ls_ff")

        self.cue_proj = Dense(self.d, use_bias=True, name=f"{self.name}_cue_proj") if self.use_tokens else None
        self.proj_out = Conv2D(self.c_out, 1, use_bias=False, name=f"{self.name}_proj_out")
        super().build(input_shape)

    def _token_desc(self, tokens):
        return tf.reduce_mean(tokens, axis=1)

    def _mha(self, q, k, v, h, d):
        b = tf.shape(q)[0]
        tq = tf.shape(q)[1]
        tk = tf.shape(k)[1]

        def _sq(t):
            return tf.transpose(tf.reshape(t, [b, tq, h, d]), [0, 2, 1, 3])

        def _skv(t):
            return tf.transpose(tf.reshape(t, [b, tk, h, d]), [0, 2, 1, 3])

        qh = _sq(q)
        kh = _skv(k)
        vh = _skv(v)
        logits = tf.einsum("bhqd,bhkd->bhqk", qh, kh) * (1.0 / tf.sqrt(tf.cast(d, tf.float32)))
        a = tf.nn.softmax(logits, axis=-1)
        y = tf.einsum("bhqk,bhkd->bhqd", a, vh)
        return tf.reshape(tf.transpose(y, [0, 2, 1, 3]), [b, tq, h * d])

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
            tokens = inputs[1] if len(inputs) > 1 else None
        else:
            x = inputs
            tokens = None

        if tokens is not None and tokens.shape.rank == 4:
            tokens = tf.squeeze(tokens, axis=-1)

        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]

        t = tf.reshape(x, [b, h * w, c])
        th = self.proj_in(t)

        mem0 = tf.tile(tf.expand_dims(self.mem, 0), [b, 1, 1])
        if self.use_tokens and (tokens is not None) and (self.cue_proj is not None):
            mem0 = mem0 + self.cue_proj(self._token_desc(tokens))[:, None, :]

        m1 = self.norm_mem1(mem0)
        t1 = self.norm_tok1(th)
        q, k, v = self.q_proj(m1), self.k_proj(t1), self.v_proj(t1)
        d = tf.shape(q)[-1] // self.num_heads
        mem = mem0 + self.ls_r1(self._mha(q, k, v, self.num_heads, d))

        m2 = self.norm_mem2(mem)
        t2 = self.norm_tok2(th)
        q, k, v = self.q_proj(t2), self.k_proj(m2), self.v_proj(m2)
        d = tf.shape(q)[-1] // self.num_heads
        tok = th + self.ls_w1(self._mha(q, k, v, self.num_heads, d))
        tok = tok + self.ls_ff(self.ffn(self.ln_ff(tok)))

        y = tf.reshape(self.o_proj(tok), [b, h, w, self.d])
        return self.proj_out(y)


@register_serializable(name="TCSM")
class TCSM(L.Layer):
    def __init__(self, ratio=16, spatial_kernels=(3, 5, 7), dilations=(1, 2, 3), use_tokens=True, **kwargs):
        super().__init__(**kwargs)
        self.ratio = int(ratio)
        self.spatial_kernels = list(spatial_kernels)
        self.dilations = list(dilations)
        self.use_tokens = bool(use_tokens)

    def build(self, input_shape):
        x_shape = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
        c = int(x_shape[-1])
        hid = max(4, c // self.ratio)

        self.ch_fc1 = Dense(hid, activation="relu", name=f"{self.name}_ch_fc1")
        self.ch_fc2 = Dense(c, activation="linear", name=f"{self.name}_ch_fc2")

        if self.use_tokens:
            self.t_fc1 = Dense(hid, activation="relu", name=f"{self.name}_t_fc1")
            self.t_fc2 = Dense(c, activation="linear", name=f"{self.name}_t_fc2")

        self.dw_branches = []
        for i, (k, d) in enumerate(zip(self.spatial_kernels, self.dilations)):
            self.dw_branches.append(
                DepthwiseConv2D(k, padding="same", dilation_rate=d, use_bias=False, name=f"{self.name}_dw{i}_k{k}_d{d}")
            )

        sk_h = max(4, c // self.ratio)
        self.sk_fc1 = Dense(sk_h, activation="relu", name=f"{self.name}_sk_fc1")
        self.sk_fc2 = Dense(len(self.dw_branches), activation="softmax", name=f"{self.name}_sk_fc2")

        if self.use_tokens:
            self.ts_fc1 = Dense(sk_h, activation="relu", name=f"{self.name}_ts_fc1")

        self.s_proj = Conv2D(1, 1, padding="same", use_bias=True, name=f"{self.name}_s_proj")
        self.w_c = self.add_weight(name="w_c", shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        self.w_s = self.add_weight(name="w_s", shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True)
        super().build(input_shape)

    def _token_desc(self, tokens):
        return tf.reduce_mean(tokens, axis=1)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[0]
            tokens = inputs[1] if len(inputs) > 1 else None
        else:
            x = inputs
            tokens = None

        if tokens is not None and tokens.shape.rank == 4:
            tokens = tf.squeeze(tokens, axis=-1)

        c = tf.shape(x)[-1]
        gap = tf.reduce_mean(x, axis=[1, 2])
        gmp = tf.reduce_max(x, axis=[1, 2])
        ch_logits = self.ch_fc2(self.ch_fc1(tf.concat([gap, gmp], axis=-1)))
        if self.use_tokens and (tokens is not None):
            ch_logits = ch_logits + self.t_fc2(self.t_fc1(self._token_desc(tokens)))

        ch_gate = tf.nn.sigmoid(ch_logits)
        ch_gate = tf.reshape(ch_gate, (-1, 1, 1, c))

        feats = [dw(x) for dw in self.dw_branches]
        y_stack = tf.stack(feats, axis=-1)
        s = tf.reduce_mean(y_stack, axis=[1, 2, 3])
        ctrl = self.sk_fc1(s)
        if self.use_tokens and (tokens is not None):
            ctrl = ctrl + self.ts_fc1(self._token_desc(tokens))

        w = self.sk_fc2(ctrl)
        w = tf.reshape(w, (-1, 1, 1, 1, tf.shape(w)[-1]))
        y = tf.reduce_sum(y_stack * w, axis=-1)
        sp_gate = tf.nn.sigmoid(self.s_proj(y))

        return self.w_c * (x * ch_gate) + self.w_s * (x * sp_gate)


@register_serializable()
class ConfidenceGate(L.Layer):
    def build(self, input_shape):
        self.h = Conv2D(1, 1, padding="same", use_bias=True, name=f"{self.name}_conf")
        super().build(input_shape)

    def call(self, x):
        return tf.nn.sigmoid(self.h(x))


@register_serializable(name="LKAplus")
class LKAplus(L.Layer):
    def __init__(self, branches=None, reinforce=True, **kwargs):
        super().__init__(**kwargs)
        self.branches_spec = list(branches) if branches is not None else [(3, 5, 1), (5, 7, 3), (7, 9, 5)]
        self.reinforce = bool(reinforce)

    def build(self, input_shape):
        ch = int(input_shape[-1])
        self.dw1_list, self.dw2_list, self.pw_list = [], [], []
        for i, (k1, k2, d) in enumerate(self.branches_spec):
            self.dw1_list.append(DepthwiseConv2D(k1, padding="same", use_bias=False, name=f"{self.name}_b{i}_dw1"))
            self.dw2_list.append(DepthwiseConv2D(k2, padding="same", dilation_rate=d, use_bias=False, name=f"{self.name}_b{i}_dw2"))
            self.pw_list.append(Conv2D(ch, 1, padding="same", use_bias=False, name=f"{self.name}_b{i}_pw"))

        self.sk_fc1 = Dense(max(4, ch // 8), activation="relu", name=f"{self.name}_mix_fc1")
        self.sk_fc2 = Dense(len(self.branches_spec), activation="softmax", name=f"{self.name}_mix_fc2")

        if self.reinforce:
            self.edge_dw = DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{self.name}_edge_dw")
            self.edge_proj = Conv2D(1, 1, padding="same", use_bias=True, name=f"{self.name}_edge_proj")
            self.conf_gate = ConfidenceGate(name=f"{self.name}_conf")
            self.r_alpha = self.add_weight(
                name="r_alpha", shape=(), initializer=tf.keras.initializers.Constant(1.0), trainable=True
            )
        super().build(input_shape)

    def call(self, x):
        feats = [pw(dw2(dw1(x))) for dw1, dw2, pw in zip(self.dw1_list, self.dw2_list, self.pw_list)]
        y_stack = tf.stack(feats, axis=-1)
        s = tf.reduce_mean(y_stack, axis=[1, 2, 3])
        w = self.sk_fc2(self.sk_fc1(s))
        w = tf.reshape(w, (-1, 1, 1, 1, tf.shape(w)[-1]))
        y = tf.reduce_sum(y_stack * w, axis=-1)

        if self.reinforce:
            e = tf.math.abs(self.edge_dw(x))
            r = tf.nn.sigmoid(self.edge_proj(e))
            conf = self.conf_gate(x)
            y = y + self.r_alpha * r * (1.0 - conf)

        return x * tf.nn.sigmoid(y)


@register_serializable()
class SinCos2DPosEnc(L.Layer):
    def call(self, x):
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]
        c_static = x.shape[-1] or c
        half = tf.cast(c_static // 2, tf.int32)

        yy = tf.linspace(0.0, 1.0, h)
        xx = tf.linspace(0.0, 1.0, w)
        yy, xx = tf.meshgrid(yy, xx, indexing="ij")

        n = tf.cast(tf.maximum(1, half // 2), tf.int32)
        freqs = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(10000.0), n))

        def emb(coord):
            coord = tf.expand_dims(coord, -1) / freqs
            return tf.concat([tf.sin(coord), tf.cos(coord)], axis=-1)

        pe = tf.concat([emb(yy), emb(xx)], axis=-1)
        pad = tf.maximum(0, c - tf.shape(pe)[-1])
        pe = tf.pad(pe, [[0, 0], [0, 0], [0, pad]])
        return x + tf.cast(tf.expand_dims(pe, 0), x.dtype)


@register_serializable(name="StableSkipGate")
class StableSkipGate(L.Layer):
    def build(self, input_shapes):
        ch_s = int(input_shapes[0][-1])
        self.theta = Conv2D(ch_s, 1, use_bias=False, name=f"{self.name}_theta")
        self.phi = Conv2D(ch_s, 1, use_bias=False, name=f"{self.name}_phi")
        self.psi = Conv2D(1, 1, use_bias=True, name=f"{self.name}_psi")
        super().build(input_shapes)

    def call(self, inputs):
        skip, dec = inputs
        a = tf.nn.relu(self.theta(skip) + self.phi(dec))
        g_sp = tf.nn.sigmoid(self.psi(a))
        return skip * (0.5 + 0.5 * g_sp)


@register_serializable()
class EdgeHead(L.Layer):
    def build(self, input_shape):
        self.dw = DepthwiseConv2D(3, padding="same", use_bias=False, name=f"{self.name}_dw")
        self.proj = Conv2D(1, 1, padding="same", use_bias=True, name=f"{self.name}_proj")
        super().build(input_shape)

    def call(self, x):
        return tf.nn.sigmoid(self.proj(tf.math.abs(self.dw(x))))


@register_serializable()
class EdgeRefine(L.Layer):
    def __init__(self, num_classes=1, alpha_init=0.05, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.alpha_init = float(alpha_init)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(self.num_classes,),
            initializer=tf.keras.initializers.Constant(self.alpha_init),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        logits, edge, conf = inputs
        delta = edge * (1.0 - conf)
        delta = tf.tile(delta, [1, 1, 1, tf.shape(logits)[-1]])
        return logits + self.alpha * delta
