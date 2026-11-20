"""SAM2 (Hiera image encoder + box-prompted mask decoder) in plain JAX.

A port of the MLX single-image training forward pass in qgis-segment
(``streuobst.inference.backends.sam2_mlx``, ``Sam2Trainable.__call__``):

    images (B, S, S, 3) preprocessed, boxes (B, 4) XYXY in input pixels
        -> low-res mask logits (B, 1, S/4, S/4)

Parameters are a nested dict/list pytree of float32 jax arrays and the model
is a set of pure functions of ``(params, config, inputs)``. LoRA adapters for
the trunk's qkv projections live in a second pytree (``init_lora``), so a
training step differentiates with respect to that tree only.

Everything is float32, static-shape and channels-last. The op set is kept
small on purpose: one strided convolution (the patch embedding), matmuls,
reshapes/transposes, concatenations and slices, reductions, exp/erf/sin/cos. The 1x1
convolutions are matmuls, the 2x2/stride-2 max pooling and transposed
convolutions are reshapes plus a reduction/einsum, and every interpolation
is a matmul with a constant matrix computed at trace time.
"""

import functools
import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = [
    "HIERA_CONFIGS",
    "Sam2Config",
    "block_specs",
    "config_from_weights",
    "forward",
    "init_lora",
    "init_params",
    "interp_matrix",
    "load_params",
]

# Architecture facts that parameter shapes do not determine, keyed by trunk
# depth (sam2.1 hydra configs; same table as the MLX port).
HIERA_CONFIGS: dict[int, dict[str, Any]] = {
    12: {  # hiera-tiny
        "num_heads": 1,
        "stages": (1, 2, 7, 2),
        "global_att_blocks": (5, 7, 9),
        "pos_embed_bkg_size": (7, 7),
        "window_spec": (8, 4, 14, 7),
    },
    16: {  # hiera-small
        "num_heads": 1,
        "stages": (1, 2, 11, 2),
        "global_att_blocks": (7, 10, 13),
        "pos_embed_bkg_size": (7, 7),
        "window_spec": (8, 4, 14, 7),
    },
    24: {  # hiera-base-plus
        "num_heads": 2,
        "stages": (2, 3, 16, 3),
        "global_att_blocks": (12, 16, 20),
        "pos_embed_bkg_size": (14, 14),
        "window_spec": (8, 4, 14, 7),
    },
    48: {  # hiera-large
        "num_heads": 2,
        "stages": (2, 6, 36, 4),
        "global_att_blocks": (23, 33, 43),
        "pos_embed_bkg_size": (14, 14),
        "window_spec": (8, 4, 16, 8),
    },
}

_Q_POOL = 3  # number of stage transitions that pool (all three)
_PATCH_STRIDE = 4
_EMBED_STRIDE = 16  # image pixels per image-embedding cell


@dataclass(frozen=True)
class Sam2Config:
    """Static architecture description. Defaults are sam2.1 hiera-tiny."""

    image_size: int = 1024
    embed_dim: int = 96
    num_heads: int = 1
    stages: tuple[int, ...] = (1, 2, 7, 2)
    global_att_blocks: tuple[int, ...] = (5, 7, 9)
    window_spec: tuple[int, ...] = (8, 4, 14, 7)
    pos_embed_bkg_size: tuple[int, int] = (7, 7)
    d_model: int = 256  # neck / prompt / decoder width
    decoder_depth: int = 2
    decoder_heads: int = 8
    decoder_mlp_dim: int = 2048
    num_mask_tokens: int = 4
    # Global attention on Palladium's flash-attention kernels (see
    # examples/flash_attention.py) instead of a score matrix written out in
    # full. Needs the `metal` platform; the kernels match the plain form.
    fused_attention: bool = False

    def __post_init__(self):
        if self.image_size % 32 != 0:
            # 4x patch stride, three 2x poolings, and the 8x8 window
            # positional embedding tiled over the S/4 token grid.
            raise ValueError(f"image_size must be a multiple of 32, got {self.image_size}")
        if len(self.stages) != 4 or len(self.window_spec) != 4:
            raise ValueError("stages and window_spec must have four entries")

    @property
    def embed_hw(self) -> int:
        """Side of the image embedding the mask decoder attends over."""
        return self.image_size // _EMBED_STRIDE

    @property
    def mask_hw(self) -> int:
        """Side of the low-res mask logits."""
        return self.image_size // _PATCH_STRIDE


@dataclass(frozen=True)
class BlockSpec:
    dim: int
    dim_out: int
    num_heads: int
    q_pool: bool
    window_size: int  # 0 = global attention
    fused: bool = False  # attention on the flash-attention kernels


def block_specs(config: Sam2Config) -> list[BlockSpec]:
    """Per-block layout of the Hiera trunk (mirrors ``hieradet.Hiera``)."""
    stage_ends = _stage_ends(config)
    q_pool_blocks = [e + 1 for e in stage_ends[:-1]][:_Q_POOL]
    specs = []
    dim, heads, cur_stage = config.embed_dim, config.num_heads, 1
    for i in range(sum(config.stages)):
        dim_out = dim
        # Lags by a block: the first block of a stage still uses the previous
        # stage's window size.
        window = 0 if i in config.global_att_blocks else config.window_spec[cur_stage - 1]
        if i - 1 in stage_ends:
            dim_out, heads, cur_stage = dim * 2, heads * 2, cur_stage + 1
        pooled = i in q_pool_blocks
        fused = config.fused_attention and window == 0 and not pooled
        specs.append(BlockSpec(dim, dim_out, heads, pooled, window, fused))
        dim = dim_out
    return specs


def _stage_ends(config: Sam2Config) -> list[int]:
    return [sum(config.stages[: i + 1]) - 1 for i in range(len(config.stages))]


# --------------------------------------------------------------------------
# Constant interpolation matrices (numpy, evaluated at trace time)
# --------------------------------------------------------------------------


def _cubic(t: float, a: float = -0.75) -> float:
    t = abs(t)
    if t <= 1.0:
        return (a + 2.0) * t**3 - (a + 3.0) * t**2 + 1.0
    if t < 2.0:
        return a * t**3 - 5.0 * a * t**2 + 8.0 * a * t - 4.0 * a
    return 0.0


def interp_matrix(out_size: int, in_size: int, kind: str) -> np.ndarray:
    """(out_size, in_size) float32 matrix M with ``M @ x`` equal to torch's
    ``F.interpolate(x, mode=kind, align_corners=False)`` along one axis:
    half-pixel centres, source index clamped to the border, bicubic with
    A = -0.75. Same construction as the MLX port's ``interp_matrix``."""
    m = np.zeros((out_size, in_size), dtype=np.float32)
    scale = in_size / out_size
    for i in range(out_size):
        src = (i + 0.5) * scale - 0.5
        f = math.floor(src)
        t = src - f
        if kind == "cubic":
            taps = [(f + k, _cubic(t - k)) for k in range(-1, 3)]
        elif kind == "linear":
            taps = [(f, 1.0 - t), (f + 1, t)]
        else:
            raise ValueError(f"unknown interpolation kind: {kind}")
        for idx, w in taps:
            m[i, min(max(idx, 0), in_size - 1)] += w
    return m


def resize_2d(x, size: tuple[int, int], kind: str = "linear"):
    """Interpolate the trailing two dims of ``(..., H, W)`` to ``size`` as two
    matmuls with constant matrices (torch ``align_corners=False`` semantics)."""
    wh = jnp.asarray(interp_matrix(size[0], x.shape[-2], kind))
    ww = jnp.asarray(interp_matrix(size[1], x.shape[-1], kind))
    return jnp.matmul(jnp.matmul(wh, x), ww.T)


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


def _linear(p, x):
    return jnp.matmul(x, p["weight"].T) + p["bias"]


def _layer_norm(p, x, eps):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * lax.rsqrt(var + eps) * p["weight"] + p["bias"]


def _gelu(x):
    # Exact (erf) GELU, as torch's nn.GELU() and mlx's nn.gelu.
    return 0.5 * x * (1.0 + lax.erf(x * (1.0 / math.sqrt(2.0))))


def _attend(q, k, v, scale):
    """Scaled dot-product attention over (B, heads, L, head_dim) arrays."""
    scores = jnp.matmul(q * scale, jnp.swapaxes(k, -1, -2))
    scores = scores - lax.stop_gradient(jnp.max(scores, axis=-1, keepdims=True))
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.matmul(weights, v)


def _max_pool_2x2(x):
    """2x2 max pooling with stride 2 on (B, H, W, C), H and W even."""
    b, h, w, c = x.shape
    return jnp.max(x.reshape(b, h // 2, 2, w // 2, 2, c), axis=(2, 4))


def _upsample_nearest_2x(x):
    b, h, w, c = x.shape
    return jnp.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c)).reshape(
        b, 2 * h, 2 * w, c
    )


def _conv_transpose_2x2(p, x):
    """ConvTranspose2d(kernel 2, stride 2): the taps do not overlap, so every
    input cell writes its own 2x2 output block. ``weight`` is torch's
    (C_in, C_out, 2, 2)."""
    b, h, w, _ = x.shape
    y = jnp.einsum("bhwi,ioyx->bhywxo", x, p["weight"])
    return y.reshape(b, 2 * h, 2 * w, -1) + p["bias"]


# --------------------------------------------------------------------------
# Hiera trunk
# --------------------------------------------------------------------------


def _window_partition(x, ws: int):
    b, h, w, c = x.shape
    pad_h, pad_w = (ws - h % ws) % ws, (ws - w % ws) % ws
    # Zero-pad bottom/right by concatenation: its gradient is a plain slice,
    # where the gradient of jnp.pad lowers to a pad with negative widths.
    if pad_h:
        x = jnp.concatenate([x, jnp.zeros((b, pad_h, w, c), x.dtype)], axis=1)
    if pad_w:
        x = jnp.concatenate([x, jnp.zeros((b, h + pad_h, pad_w, c), x.dtype)], axis=2)
    hp, wp = h + pad_h, w + pad_w
    x = x.reshape(b, hp // ws, ws, wp // ws, ws, c)
    return x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, c), (hp, wp)


def _window_unpartition(windows, ws: int, pad_hw, hw):
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // ((hp // ws) * (wp // ws))
    x = windows.reshape(b, hp // ws, wp // ws, ws, ws, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, -1)
    if hp > h or wp > w:
        x = x[:, :h, :w, :]
    return x


def _lora_qkv(p, lora, x):
    """qkv projection plus the low-rank update on its q and v output slices
    (``LoRAQkv`` in the MLX port: no alpha/rank scaling, k untouched)."""
    out = _linear(p, x)
    if lora is None:
        return out
    q_up = jnp.matmul(jnp.matmul(x, lora["lora_a_q"].T), lora["lora_b_q"].T)
    v_up = jnp.matmul(jnp.matmul(x, lora["lora_a_v"].T), lora["lora_b_v"].T)
    return out + jnp.concatenate([q_up, jnp.zeros_like(q_up), v_up], axis=-1)


@functools.cache
def _fused_attention(shape: tuple[int, int, int, int]):
    """Attention over (batch, length, heads, head_dim) arrays on the
    flash-attention kernels, with their backward kernels as its gradient.

    One threadgroup per tile of 64 queries keeps the softmax in registers, so
    the (length, length) scores never reach memory. The backward tiles
    shrink for wide heads to stay within Metal's threadgroup memory.
    """
    from flash_attention import attention_backward_calls, attention_call

    import palladium

    forward = attention_call(shape, tile_q=64, tile_k=16, logsumexp=True)
    dkdv, dq = attention_backward_calls(shape, rows=32 if shape[3] <= 64 else 16)

    def run(q, k, v):
        out, logsumexp = forward(q, k, v)
        return out, out, logsumexp  # the output and the logsumexp are residuals

    def backward(q, k, v, out, logsumexp, d_out):
        delta = jnp.sum(d_out * out, axis=-1).transpose(0, 2, 1)
        dk, dv = dkdv(q, k, v, d_out, logsumexp, delta)
        return dq(q, k, v, d_out, logsumexp, delta), dk, dv

    return palladium.with_vjp(run, backward, residuals=2)


def _multiscale_attention(p, spec: BlockSpec, x, lora):
    b, h, w, _ = x.shape
    heads = spec.num_heads
    qkv = _lora_qkv(p["qkv"], lora, x).reshape(b, h * w, 3, heads, -1)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    # The kernels take whole query tiles and head widths in SIMD fragments.
    if spec.fused and (h * w) % 64 == 0 and q.shape[-1] % 16 == 0:
        out = _fused_attention(q.shape)(q, k, v)
        return _linear(p["proj"], out.reshape(b, h, w, -1))
    if spec.q_pool:
        q = _max_pool_2x2(q.reshape(b, h, w, -1))
        h, w = q.shape[1:3]
        q = q.reshape(b, h * w, heads, -1)
    out = _attend(
        q.transpose(0, 2, 1, 3),
        k.transpose(0, 2, 1, 3),
        v.transpose(0, 2, 1, 3),
        (spec.dim_out // heads) ** -0.5,
    )
    return _linear(p["proj"], out.transpose(0, 2, 1, 3).reshape(b, h, w, -1))


def _multiscale_block(p, spec: BlockSpec, x, lora):
    shortcut = x
    x = _layer_norm(p["norm1"], x, 1e-6)
    if spec.dim != spec.dim_out:
        shortcut = _linear(p["proj"], x)
        if spec.q_pool:
            shortcut = _max_pool_2x2(shortcut)

    ws = spec.window_size
    h, w = x.shape[1:3]
    pad_hw = (h, w)
    if ws > 0:
        x, pad_hw = _window_partition(x, ws)
    x = _multiscale_attention(p["attn"], spec, x, lora)
    if spec.q_pool:
        ws = spec.window_size // 2
        h, w = shortcut.shape[1:3]
        pad_hw = (h + (ws - h % ws) % ws, w + (ws - w % ws) % ws) if ws > 0 else (h, w)
    if spec.window_size > 0:
        x = _window_unpartition(x, ws, pad_hw, (h, w))

    x = shortcut + x
    y = _layer_norm(p["norm2"], x, 1e-6)
    return x + _linear(p["mlp"][1], _gelu(_linear(p["mlp"][0], y)))


def _hiera_pos_embed(p, hw: tuple[int, int]):
    """Background embedding bicubic-interpolated to the token grid, plus the
    window embedding tiled over it; (1, h, w, C)."""
    h, w = hw
    bkg, window = p["pos_embed"], p["pos_embed_window"]  # (1, C, ., .)
    wh = jnp.asarray(interp_matrix(h, bkg.shape[2], "cubic"))
    ww = jnp.asarray(interp_matrix(w, bkg.shape[3], "cubic"))
    pos = jnp.matmul(jnp.matmul(wh, bkg), ww.T)
    pos = pos + jnp.tile(window, (1, 1, h // window.shape[2], w // window.shape[3]))
    return pos.transpose(0, 2, 3, 1)


def hiera_trunk(p, config: Sam2Config, images, lora=None):
    """(B, S, S, 3) -> the four stage outputs, highest resolution first."""
    x = lax.conv_general_dilated(
        images,
        p["patch_embed"]["weight"],  # HWIO, 7x7
        window_strides=(_PATCH_STRIDE, _PATCH_STRIDE),
        padding=((3, 3), (3, 3)),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    x = x + p["patch_embed"]["bias"]
    x = x + _hiera_pos_embed(p, x.shape[1:3])
    stage_ends = _stage_ends(config)
    outputs = []
    for i, spec in enumerate(block_specs(config)):
        block_lora = None if lora is None else lora.get(str(i))
        x = _multiscale_block(p["blocks"][i], spec, x, block_lora)
        if i in stage_ends:
            outputs.append(x)
    return outputs


# --------------------------------------------------------------------------
# FPN neck
# --------------------------------------------------------------------------


def fpn_neck(p, xs):
    """Stage outputs -> d_model-channel levels in the same order. 1x1 lateral
    convolutions; only level 2 receives a top-down (nearest 2x) contribution,
    from level 3 (``fpn_top_down_levels=[2, 3]``)."""
    n = len(p) - 1
    out = [None] * len(p)
    prev = None
    for i in range(n, -1, -1):
        lateral = _linear(p[n - i], xs[i])
        prev = lateral + _upsample_nearest_2x(prev) if i in (2, 3) and prev is not None else lateral
        out[i] = prev
    return out


# --------------------------------------------------------------------------
# Prompt encoder
# --------------------------------------------------------------------------


def _pe_encoding(gaussian, coords):
    """Random-Fourier features of coordinates in [0, 1]; (..., 2) -> (..., C)."""
    coords = jnp.matmul(2.0 * coords - 1.0, gaussian) * (2.0 * math.pi)
    return jnp.concatenate([jnp.sin(coords), jnp.cos(coords)], axis=-1)


def dense_pe(p, config: Sam2Config):
    """(1, h, w, C) positional encoding of the image-embedding grid."""
    n = config.embed_hw
    centres = (jnp.arange(n, dtype=jnp.float32) + 0.5) / n
    grid = jnp.stack(
        [jnp.broadcast_to(centres[None, :], (n, n)), jnp.broadcast_to(centres[:, None], (n, n))],
        axis=-1,
    )
    return _pe_encoding(p["gaussian"], grid)[None]


def encode_boxes(p, config: Sam2Config, boxes):
    """Boxes (B, 4) XYXY -> sparse prompt tokens (B, 3, C): the two corners as
    points labelled 2 and 3, then the padding point (label -1), which is the
    ``not_a_point`` embedding alone."""
    b = boxes.shape[0]
    corners = boxes.reshape(b, 2, 2) + 0.5  # pixel centres
    pe = _pe_encoding(p["gaussian"], corners * (1.0 / config.image_size))
    pe = pe + p["point_embeddings"][2:4]
    pad = jnp.broadcast_to(p["not_a_point_embed"], (b, 1, pe.shape[-1]))
    return jnp.concatenate([pe, pad], axis=1)


# --------------------------------------------------------------------------
# Mask decoder
# --------------------------------------------------------------------------


def _attention(p, heads: int, q, k, v):
    q, k, v = _linear(p["q_proj"], q), _linear(p["k_proj"], k), _linear(p["v_proj"], v)
    b, n, c = q.shape
    head_dim = c // heads

    def split(x):
        return x.reshape(b, x.shape[1], heads, head_dim).transpose(0, 2, 1, 3)

    out = _attend(split(q), split(k), split(v), head_dim**-0.5)
    return _linear(p["out_proj"], out.transpose(0, 2, 1, 3).reshape(b, n, c))


def _two_way_block(p, heads: int, queries, keys, query_pe, key_pe, skip_first_layer_pe: bool):
    if skip_first_layer_pe:
        queries = _attention(p["self_attn"], heads, queries, queries, queries)
    else:
        q = queries + query_pe
        queries = queries + _attention(p["self_attn"], heads, q, q, queries)
    queries = _layer_norm(p["norm1"], queries, 1e-5)

    q, k = queries + query_pe, keys + key_pe
    queries = queries + _attention(p["cross_attn_token_to_image"], heads, q, k, keys)
    queries = _layer_norm(p["norm2"], queries, 1e-5)

    mlp = _linear(p["mlp"][1], jnp.maximum(_linear(p["mlp"][0], queries), 0.0))
    queries = _layer_norm(p["norm3"], queries + mlp, 1e-5)

    q, k = queries + query_pe, keys + key_pe
    keys = keys + _attention(p["cross_attn_image_to_token"], heads, k, q, queries)
    keys = _layer_norm(p["norm4"], keys, 1e-5)
    return queries, keys


def mask_decoder(p, config: Sam2Config, image_embed, image_pe, sparse, dense, high_res):
    """Single-mask (token 0) logits (B, 1, 4h, 4w) for an (B, h, w, C) image
    embedding; the training-mode output of the MLX decoder for
    ``multimask_output=False``. The IoU and object-score heads and the other
    three hypernetwork MLPs do not reach this output and are not evaluated."""
    heads = config.decoder_heads
    output_tokens = jnp.concatenate([p["obj_score_token"], p["iou_token"], p["mask_tokens"]])
    b = sparse.shape[0]
    tokens = jnp.concatenate(
        [jnp.broadcast_to(output_tokens[None], (b, *output_tokens.shape)), sparse], axis=1
    )

    src = image_embed + dense
    _, h, w, c = src.shape
    keys = src.reshape(b, h * w, c)
    key_pe = jnp.broadcast_to(image_pe, (b, h, w, c)).reshape(b, h * w, c)

    t = p["transformer"]
    queries = tokens
    for i, layer in enumerate(t["layers"]):
        queries, keys = _two_way_block(layer, heads, queries, keys, tokens, key_pe, i == 0)
    q, k = queries + tokens, keys + key_pe
    queries = queries + _attention(t["final_attn_token_to_image"], heads, q, k, keys)
    queries = _layer_norm(t["norm_final_attn"], queries, 1e-5)

    mask_token_out = queries[:, 2]  # after the object-score and IoU tokens

    feat_s0, feat_s1 = high_res
    up = p["output_upscaling"]
    x = _conv_transpose_2x2(up["dc1"], keys.reshape(b, h, w, c)) + feat_s1
    x = _gelu(_layer_norm(up["ln1"], x, 1e-6))
    x = _gelu(_conv_transpose_2x2(up["dc2"], x) + feat_s0)

    hyper = mask_token_out
    layers = p["hypernetwork_mlp0"]
    for i, layer in enumerate(layers):
        hyper = _linear(layer, hyper)
        if i < len(layers) - 1:
            hyper = jnp.maximum(hyper, 0.0)

    _, hh, ww, cc = x.shape
    masks = jnp.matmul(x.reshape(b, hh * ww, cc), hyper[:, :, None])
    return masks.reshape(b, 1, hh, ww)


# --------------------------------------------------------------------------
# Full forward
# --------------------------------------------------------------------------


def forward(params, config: Sam2Config, images, boxes, lora=None, return_intermediates=False):
    """The SAM2 training forward pass.

    Args:
        params: Pytree from :func:`load_params` or :func:`init_params`.
        config: Static architecture description.
        images: (B, S, S, 3) float32, preprocessed (ImageNet-normalized).
        boxes: (B, 4) float32 XYXY in input pixels (0..S).
        lora: Optional pytree from :func:`init_lora`, ``{block index: {...}}``.
        return_intermediates: Also return a dict of intermediate tensors.

    Returns:
        Mask logits (B, 1, S/4, S/4), and the intermediates dict if requested.
    """
    if images.shape[1:] != (config.image_size, config.image_size, 3):
        raise ValueError(f"expected images (B, {config.image_size}, {config.image_size}, 3)")
    stages = hiera_trunk(params["trunk"], config, images, lora)
    feats = fpn_neck(params["neck"], stages)
    dec = params["mask_decoder"]
    feat_s0 = _linear(dec["conv_s0"], feats[0])
    feat_s1 = _linear(dec["conv_s1"], feats[1])
    image_embed = feats[2] + params["no_mem_embed"]

    prompt = params["prompt_encoder"]
    sparse = encode_boxes(prompt, config, boxes)
    image_pe = dense_pe(prompt, config)
    dense = prompt["no_mask_embed"]  # broadcasts over (B, h, w, C)
    logits = mask_decoder(dec, config, image_embed, image_pe, sparse, dense, (feat_s0, feat_s1))
    if not return_intermediates:
        return logits
    inter = {f"trunk_{i}": s for i, s in enumerate(stages)}
    inter |= {f"neck_{i}": f for i, f in enumerate(feats)}
    inter |= {
        "feat_s0": feat_s0,
        "feat_s1": feat_s1,
        "image_embed": image_embed,
        "sparse": sparse,
        "dense_pe": image_pe,
        "logits": logits,
    }
    return logits, inter


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


def config_from_weights(weights, image_size: int = 1024) -> Sam2Config:
    """Infer the config from a flat checkpoint dict (torch parameter names)."""
    prefix = "image_encoder.trunk.blocks."
    depth = 1 + max(int(k[len(prefix) :].split(".")[0]) for k in weights if k.startswith(prefix))
    if depth not in HIERA_CONFIGS:
        raise ValueError(f"unknown Hiera depth {depth}; expected one of {list(HIERA_CONFIGS)}")
    return Sam2Config(
        image_size=image_size,
        embed_dim=int(weights["image_encoder.trunk.patch_embed.proj.weight"].shape[0]),
        d_model=int(weights["no_mem_embed"].shape[-1]),
        num_mask_tokens=int(weights["sam_mask_decoder.mask_tokens.weight"].shape[0]),
        **HIERA_CONFIGS[depth],
    )


def load_params(weights, config: Sam2Config):
    """Build the parameter pytree from a flat ``{torch name: array}`` mapping
    (the ``.npz`` written by ``convert.py``).

    Linear, LayerNorm and embedding weights are used as stored. Convolution
    weights are re-laid out once here: the patch embedding to HWIO, the 1x1
    convolutions to (C_out, C_in) matrices; the transposed convolutions keep
    torch's (C_in, C_out, 2, 2). Weights that the training forward never
    reads (mask-prompt branch, IoU and object-score heads, hypernetwork MLPs
    1-3) are left out.
    """

    def arr(name):
        return jnp.asarray(np.asarray(weights[name], dtype=np.float32))

    def wb(prefix):
        return {"weight": arr(prefix + ".weight"), "bias": arr(prefix + ".bias")}

    def conv1x1(prefix):
        return {"weight": arr(prefix + ".weight")[:, :, 0, 0], "bias": arr(prefix + ".bias")}

    def attention(prefix):
        return {n: wb(f"{prefix}.{n}") for n in ("q_proj", "k_proj", "v_proj", "out_proj")}

    tp = "image_encoder.trunk"
    blocks = []
    for i, spec in enumerate(block_specs(config)):
        bp = f"{tp}.blocks.{i}"
        block = {
            "norm1": wb(f"{bp}.norm1"),
            "attn": {"qkv": wb(f"{bp}.attn.qkv"), "proj": wb(f"{bp}.attn.proj")},
            "norm2": wb(f"{bp}.norm2"),
            "mlp": [wb(f"{bp}.mlp.layers.0"), wb(f"{bp}.mlp.layers.1")],
        }
        if spec.dim != spec.dim_out:
            block["proj"] = wb(f"{bp}.proj")
        blocks.append(block)
    trunk = {
        "pos_embed": arr(f"{tp}.pos_embed"),
        "pos_embed_window": arr(f"{tp}.pos_embed_window"),
        "patch_embed": {
            "weight": arr(f"{tp}.patch_embed.proj.weight").transpose(2, 3, 1, 0),
            "bias": arr(f"{tp}.patch_embed.proj.bias"),
        },
        "blocks": blocks,
    }

    neck = [conv1x1(f"image_encoder.neck.convs.{i}.conv") for i in range(len(config.stages))]

    pp = "sam_prompt_encoder"
    prompt = {
        "gaussian": arr(f"{pp}.pe_layer.positional_encoding_gaussian_matrix"),
        "point_embeddings": jnp.concatenate(
            [arr(f"{pp}.point_embeddings.{i}.weight") for i in range(4)]
        ),
        "not_a_point_embed": arr(f"{pp}.not_a_point_embed.weight")[0],
        "no_mask_embed": arr(f"{pp}.no_mask_embed.weight")[0],
    }

    dp = "sam_mask_decoder"
    layers = []
    for i in range(config.decoder_depth):
        lp = f"{dp}.transformer.layers.{i}"
        layers.append(
            {
                "self_attn": attention(f"{lp}.self_attn"),
                "norm1": wb(f"{lp}.norm1"),
                "cross_attn_token_to_image": attention(f"{lp}.cross_attn_token_to_image"),
                "norm2": wb(f"{lp}.norm2"),
                "mlp": [wb(f"{lp}.mlp.layers.0"), wb(f"{lp}.mlp.layers.1")],
                "norm3": wb(f"{lp}.norm3"),
                "norm4": wb(f"{lp}.norm4"),
                "cross_attn_image_to_token": attention(f"{lp}.cross_attn_image_to_token"),
            }
        )
    decoder = {
        "transformer": {
            "layers": layers,
            "final_attn_token_to_image": attention(f"{dp}.transformer.final_attn_token_to_image"),
            "norm_final_attn": wb(f"{dp}.transformer.norm_final_attn"),
        },
        "obj_score_token": arr(f"{dp}.obj_score_token.weight"),
        "iou_token": arr(f"{dp}.iou_token.weight"),
        "mask_tokens": arr(f"{dp}.mask_tokens.weight"),
        "output_upscaling": {
            "dc1": wb(f"{dp}.output_upscaling.0"),
            "ln1": wb(f"{dp}.output_upscaling.1"),
            "dc2": wb(f"{dp}.output_upscaling.3"),
        },
        "conv_s0": conv1x1(f"{dp}.conv_s0"),
        "conv_s1": conv1x1(f"{dp}.conv_s1"),
        "hypernetwork_mlp0": [wb(f"{dp}.output_hypernetworks_mlps.0.layers.{i}") for i in range(3)],
    }
    return {
        "no_mem_embed": arr("no_mem_embed").reshape(-1),
        "trunk": trunk,
        "neck": neck,
        "prompt_encoder": prompt,
        "mask_decoder": decoder,
    }


def init_params(config: Sam2Config, seed: int = 0):
    """Randomly initialized parameters with the layout of :func:`load_params`
    (for shape tests and smoke runs without a checkpoint; not a training
    initialization)."""
    rng = np.random.default_rng(seed)

    def normal(*shape, std=0.02):
        return jnp.asarray(rng.normal(0.0, std, shape).astype(np.float32))

    def linear(n_in, n_out):
        std = 1.0 / math.sqrt(n_in)
        return {"weight": normal(n_out, n_in, std=std), "bias": normal(n_out)}

    def norm(n):
        return {"weight": jnp.ones((n,), jnp.float32), "bias": jnp.zeros((n,), jnp.float32)}

    def attention(dim, internal):
        return {
            "q_proj": linear(dim, internal),
            "k_proj": linear(dim, internal),
            "v_proj": linear(dim, internal),
            "out_proj": linear(internal, dim),
        }

    c, d = config.embed_dim, config.d_model
    blocks = []
    specs = block_specs(config)
    for spec in specs:
        block = {
            "norm1": norm(spec.dim),
            "attn": {
                "qkv": linear(spec.dim, 3 * spec.dim_out),
                "proj": linear(spec.dim_out, spec.dim_out),
            },
            "norm2": norm(spec.dim_out),
            "mlp": [linear(spec.dim_out, 4 * spec.dim_out), linear(4 * spec.dim_out, spec.dim_out)],
        }
        if spec.dim != spec.dim_out:
            block["proj"] = linear(spec.dim, spec.dim_out)
        blocks.append(block)
    trunk = {
        "pos_embed": normal(1, c, *config.pos_embed_bkg_size),
        "pos_embed_window": normal(1, c, config.window_spec[0], config.window_spec[0]),
        "patch_embed": {"weight": normal(7, 7, 3, c, std=0.08), "bias": normal(c)},
        "blocks": blocks,
    }
    stage_dims = [specs[e].dim_out for e in _stage_ends(config)]
    layers = [
        {
            "self_attn": attention(d, d),
            "norm1": norm(d),
            "cross_attn_token_to_image": attention(d, d // 2),
            "norm2": norm(d),
            "mlp": [linear(d, config.decoder_mlp_dim), linear(config.decoder_mlp_dim, d)],
            "norm3": norm(d),
            "norm4": norm(d),
            "cross_attn_image_to_token": attention(d, d // 2),
        }
        for _ in range(config.decoder_depth)
    ]
    decoder = {
        "transformer": {
            "layers": layers,
            "final_attn_token_to_image": attention(d, d // 2),
            "norm_final_attn": norm(d),
        },
        "obj_score_token": normal(1, d, std=1.0),
        "iou_token": normal(1, d, std=1.0),
        "mask_tokens": normal(config.num_mask_tokens, d, std=1.0),
        "output_upscaling": {
            "dc1": {"weight": normal(d, d // 4, 2, 2, std=0.05), "bias": normal(d // 4)},
            "ln1": norm(d // 4),
            "dc2": {"weight": normal(d // 4, d // 8, 2, 2, std=0.05), "bias": normal(d // 8)},
        },
        "conv_s0": linear(d, d // 8),
        "conv_s1": linear(d, d // 4),
        "hypernetwork_mlp0": [linear(d, d), linear(d, d), linear(d, d // 8)],
    }
    return {
        "no_mem_embed": normal(d),
        "trunk": trunk,
        "neck": [linear(dim, d) for dim in reversed(stage_dims)],
        "prompt_encoder": {
            "gaussian": normal(2, d // 2, std=1.0),
            "point_embeddings": normal(4, d, std=1.0),
            "not_a_point_embed": normal(d, std=1.0),
            "no_mask_embed": normal(d, std=1.0),
        },
        "mask_decoder": decoder,
    }


def init_lora(config: Sam2Config, rank: int, seed: int = 0, layers=None):
    """LoRA adapters for ``blocks[i].attn.qkv`` of every selected trunk block
    (default: all), as ``{str(i): {lora_a_q, lora_b_q, lora_a_v, lora_b_v}}``.

    Same convention as the MLX port's ``inject_lora``: A (rank, dim) drawn
    from U(-1/sqrt(dim), 1/sqrt(dim)), B (dim_out, rank) zero, so the adapted
    model starts identical to the base model.
    """
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    rng = np.random.default_rng(seed)
    specs = block_specs(config)
    lora = {}
    for i in layers if layers is not None else range(len(specs)):
        spec = specs[i]
        bound = 1.0 / math.sqrt(spec.dim)
        lora[str(i)] = {
            "lora_a_q": jnp.asarray(rng.uniform(-bound, bound, (rank, spec.dim)), jnp.float32),
            "lora_b_q": jnp.zeros((spec.dim_out, rank), jnp.float32),
            "lora_a_v": jnp.asarray(rng.uniform(-bound, bound, (rank, spec.dim)), jnp.float32),
            "lora_b_v": jnp.zeros((spec.dim_out, rank), jnp.float32),
        }
    return lora
