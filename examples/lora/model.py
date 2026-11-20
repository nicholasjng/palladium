"""Qwen2.5 decoder with LoRA adapters in plain JAX.

A port of mlx-lm's ``mlx_lm/models/qwen2.py`` and of the LoRA layer in
``mlx_lm/tuner/lora.py`` (mlx-lm 0.32.2):

    tokens (B, L) int32 -> logits (B, L, vocab)

Token embedding, pre-norm blocks of grouped-query attention with RoPE and a
causal mask plus a SwiGLU MLP, a final RMSNorm, and the LM head, which is the
transposed embedding when the checkpoint ties them. Parameters are a nested
dict/list pytree and the model is a set of pure functions of ``(params,
config, tokens)``. LoRA adapters live in a second pytree (``init_lora``), so a
training step differentiates with respect to that tree only.

Shapes are static and the op set is small: one gather (the embedding),
matmuls, reshapes/transposes, slices and concatenations, reductions, and
exp/log/logistic/rsqrt. The RoPE sine and cosine tables are float32 constants
computed at trace time. Grouped-query attention folds the query heads that
share a key/value head into the query axis, so both attention products are
plain batched matmuls without a broadcast of the keys or values.

Parameters may be float32, bfloat16 or float16; the activations take their
dtype. Half-precision follows what MLX computes on the GPU (measured against
mlx 0.32.4, see README.md): RMSNorm, RoPE, the whole attention product and
the softmax run in float32 and round their result; SiLU, the residual
additions and the RMSNorm weight stay in the parameter dtype; a matmul
accumulates in float32 and rounds once (``QwenConfig.accumulate_float32``).
For float32 parameters all of these casts vanish.
"""

import json
import math
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

__all__ = [
    "LORA_KEYS",
    "QwenConfig",
    "config_from_weights",
    "forward",
    "init_lora",
    "init_params",
    "load_params",
    "lora_layers",
]

# The linear layers of one block, in the order mlx-lm adapts them: with no
# "keys" entry in its LoRA config (the default), linear_to_lora_layers wraps
# every nn.Linear inside each of the last ``num_layers`` blocks.
LORA_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj")
_BIASED = ("q_proj", "k_proj", "v_proj")  # the Qwen2 attention projections with a bias


@dataclass(frozen=True)
class QwenConfig:
    """Architecture (defaults: Qwen2.5-0.5B) and mlx-lm's default LoRA setup."""

    vocab_size: int = 151936
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    # mlx_lm/lora.py CONFIG_DEFAULTS: num_layers 16, rank 8, scale 20.0, dropout 0.0.
    lora_num_layers: int = 16
    lora_rank: int = 8
    lora_scale: float = 20.0
    # Half-precision matmuls: upcast both operands to float32 and round the
    # result, as MLX's GEMM does. False emits the dot_general in the parameter
    # dtype, for a backend whose half-precision GEMM accumulates in float32.
    accumulate_float32: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def lora_layers(config: QwenConfig) -> list[int]:
    """Indices of the adapted blocks: the last ``lora_num_layers`` ones."""
    count = min(max(config.lora_num_layers, 0), config.num_hidden_layers)
    return list(range(config.num_hidden_layers - count, config.num_hidden_layers))


def _linear_dims(config: QwenConfig) -> dict[str, tuple[int, int]]:
    """``name -> (input_dims, output_dims)`` of the linear layers of a block."""
    dim, hidden = config.hidden_size, config.intermediate_size
    kv = config.num_key_value_heads * config.head_dim
    return {
        "q_proj": (dim, dim),
        "k_proj": (dim, kv),
        "v_proj": (dim, kv),
        "o_proj": (dim, dim),
        "gate_proj": (dim, hidden),
        "down_proj": (hidden, dim),
        "up_proj": (dim, hidden),
    }


# ---------------------------------------------------------------- layers


def _accumulator(config: QwenConfig, dtype):
    return jnp.float32 if config.accumulate_float32 else dtype


def _linear(p, x, config: QwenConfig, lora=None):
    """``x @ W + b`` with ``W`` stored (in, out), plus the LoRA branch.

    MLX: ``y + (scale * ((x @ lora_a) @ lora_b)).astype(x.dtype)`` with float32
    adapters, so the branch runs in float32 for any parameter dtype.
    """
    acc = _accumulator(config, x.dtype)
    y = x.astype(acc) @ p["weight"].astype(acc)
    if "bias" in p:
        y = y + p["bias"].astype(acc)
    y = y.astype(x.dtype)
    if lora is None:
        return y
    z = (x.astype(lora["lora_a"].dtype) @ lora["lora_a"]) @ lora["lora_b"]
    return y + (config.lora_scale * z).astype(x.dtype)


def _rms_norm(weight, x, eps: float):
    """``mx.fast.rms_norm``: normalized in float32, rounded, then weighted."""
    x32 = x.astype(jnp.float32)
    normed = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * weight


def _rope_tables(length: int, dims: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin of ``position * base**(-i / (dims/2))``, each (length, dims/2).

    The angles are formed in float32 as in ``mx.fast.rope``; their cosine and
    sine are evaluated in float64 on the host and rounded.
    """
    half = dims // 2
    freqs = np.exp(-np.arange(half, dtype=np.float32) * np.float32(math.log(base) / half))
    theta = np.arange(length, dtype=np.float32)[:, None] * freqs[None].astype(np.float32)
    theta = theta.astype(np.float64)
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def _rope(x, cos, sin):
    """Rotate (..., L, D): the pairs are ``(x[i], x[i + D/2])`` (not interleaved)."""
    half = x.shape[-1] // 2
    x32 = x.astype(jnp.float32)
    x1, x2 = x32[..., :half], x32[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(x.dtype)


def _attention(p, config: QwenConfig, x, rope, lora):
    batch, length, dim = x.shape
    heads, groups, head_dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    repeats = heads // groups
    lora = lora or {}

    q = _linear(p["q_proj"], x, config, lora.get("q_proj"))
    k = _linear(p["k_proj"], x, config, lora.get("k_proj"))
    v = _linear(p["v_proj"], x, config, lora.get("v_proj"))
    q = _rope(q.reshape(batch, length, heads, head_dim).transpose(0, 2, 1, 3), *rope)
    k = _rope(k.reshape(batch, length, groups, head_dim).transpose(0, 2, 1, 3), *rope)
    v = v.reshape(batch, length, groups, head_dim).transpose(0, 2, 1, 3)

    # Query head h attends to key/value head h // repeats. (B, G, R * L, D)
    # queries against (B, G, L, D) keys make that one batched matmul.
    q = q.astype(jnp.float32).reshape(batch, groups, repeats * length, head_dim)
    k, v = k.astype(jnp.float32), v.astype(jnp.float32)
    scores = (q * head_dim**-0.5) @ k.transpose(0, 1, 3, 2)
    scores = scores.reshape(batch, groups, repeats, length, length)
    causal = np.tril(np.ones((length, length), dtype=bool))
    scores = jnp.where(causal, scores, jnp.finfo(jnp.float32).min)
    scores = scores - jax.lax.stop_gradient(jnp.max(scores, axis=-1, keepdims=True))
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    out = weights.reshape(batch, groups, repeats * length, length) @ v
    out = out.reshape(batch, heads, length, head_dim).transpose(0, 2, 1, 3)
    out = out.reshape(batch, length, dim).astype(x.dtype)
    return _linear(p["o_proj"], out, config, lora.get("o_proj"))


def _mlp(p, config: QwenConfig, x, lora):
    lora = lora or {}
    gate = _linear(p["gate_proj"], x, config, lora.get("gate_proj"))
    up = _linear(p["up_proj"], x, config, lora.get("up_proj"))
    hidden = gate * jax.nn.sigmoid(gate) * up  # SwiGLU: silu(gate) * up
    return _linear(p["down_proj"], hidden, config, lora.get("down_proj"))


def _block(p, config: QwenConfig, x, rope, lora):
    eps = config.rms_norm_eps
    h = x + _attention(p, config, _rms_norm(p["input_layernorm"], x, eps), rope, lora)
    return h + _mlp(p, config, _rms_norm(p["post_attention_layernorm"], h, eps), lora)


def forward(params, config: QwenConfig, tokens, lora=None, return_intermediates=False):
    """Logits (B, L, vocab) in the parameter dtype for int32 ``tokens`` (B, L).

    Args:
        params: pytree from :func:`load_params` or :func:`init_params`.
        tokens: (B, L) int32 token ids.
        lora: optional pytree from :func:`init_lora`, ``{layer index as str:
            {projection: {"lora_a": (in, r), "lora_b": (r, out)}}}``. Blocks
            and projections that it does not name run without an adapter.
        return_intermediates: also return a dict with the embedding, every
            block's output, the final norm and the logits, for parity checks.
    """
    lora = lora or {}
    rope = _rope_tables(tokens.shape[1], config.head_dim, config.rope_theta)
    h = params["embed_tokens"][tokens]
    inter = {"embed": h}
    for i, layer in enumerate(params["layers"]):
        h = _block(layer, config, h, rope, lora.get(str(i)))
        inter[f"layer_{i}"] = h
    h = _rms_norm(params["norm"], h, config.rms_norm_eps)
    inter["norm"] = h
    if config.tie_word_embeddings:
        # h @ embed.T without transposing the (vocab, hidden) table.
        acc = _accumulator(config, h.dtype)
        logits = jnp.einsum("bld,vd->blv", h.astype(acc), params["embed_tokens"].astype(acc))
        logits = logits.astype(h.dtype)
    else:
        logits = _linear(params["lm_head"], h, config)
    inter["logits"] = logits
    return (logits, inter) if return_intermediates else logits


# ---------------------------------------------------------------- parameters


def config_from_weights(weights, **overrides) -> QwenConfig:
    """Read the architecture from the ``__config__`` entry ``convert.py`` writes."""
    stored = json.loads(str(weights["__config__"]))
    names = {f.name for f in fields(QwenConfig)}
    return QwenConfig(**{k: v for k, v in stored.items() if k in names} | overrides)


def _decode(array: np.ndarray) -> np.ndarray:
    """``convert.py`` stores bfloat16 arrays as their uint16 bit patterns."""
    return array.view(ml_dtypes.bfloat16) if array.dtype == np.uint16 else array


def load_params(weights, config: QwenConfig, dtype=jnp.float32):
    """Build the parameter pytree from the flat ``.npz`` of ``convert.py``.

    The ``.npz`` keeps mlx-lm's parameter names and its (out, in) linear
    weights; they are transposed to (in, out) here, once, so a linear layer is
    ``x @ W``. ``dtype`` is the dtype of every parameter (float32, bfloat16 or
    float16); bfloat16 weights convert to float32 and back without rounding.
    """

    def get(name: str, transpose: bool = False):
        array = _decode(weights[name])
        return jnp.asarray((array.T if transpose else array).astype(dtype))

    def linear(prefix: str, bias: bool):
        p = {"weight": get(f"{prefix}.weight", transpose=True)}
        if bias:
            p["bias"] = get(f"{prefix}.bias")
        return p

    layers = []
    for i in range(config.num_hidden_layers):
        prefix = f"model.layers.{i}"
        layer = {
            "input_layernorm": get(f"{prefix}.input_layernorm.weight"),
            "post_attention_layernorm": get(f"{prefix}.post_attention_layernorm.weight"),
        }
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            layer[name] = linear(f"{prefix}.self_attn.{name}", name in _BIASED)
        for name in ("gate_proj", "up_proj", "down_proj"):
            layer[name] = linear(f"{prefix}.mlp.{name}", False)
        layers.append(layer)
    params = {
        "embed_tokens": get("model.embed_tokens.weight"),
        "layers": layers,
        "norm": get("model.norm.weight"),
    }
    if not config.tie_word_embeddings:
        params["lm_head"] = linear("lm_head", False)
    return params


def init_params(config: QwenConfig, seed: int = 0, dtype=jnp.float32):
    """Random parameters with the layout of :func:`load_params`, for tests."""
    rng = np.random.default_rng(seed)

    def normal(shape, scale):
        return jnp.asarray(rng.normal(0.0, scale, shape).astype(np.float32).astype(dtype))

    def linear(name: str, dims: tuple[int, int]):
        p = {"weight": normal(dims, dims[0] ** -0.5)}
        if name in _BIASED:
            p["bias"] = normal((dims[1],), 0.1)
        return p

    def norm_weight():
        return jnp.asarray((1.0 + rng.normal(0.0, 0.1, config.hidden_size)).astype(dtype))

    layers = []
    for _ in range(config.num_hidden_layers):
        layer = {"input_layernorm": norm_weight(), "post_attention_layernorm": norm_weight()}
        layer |= {name: linear(name, dims) for name, dims in _linear_dims(config).items()}
        layers.append(layer)
    params = {
        "embed_tokens": normal((config.vocab_size, config.hidden_size), 0.5),
        "layers": layers,
        "norm": norm_weight(),
    }
    if not config.tie_word_embeddings:
        params["lm_head"] = linear("lm_head", (config.hidden_size, config.vocab_size))
    return params


def init_lora(config: QwenConfig, seed: int = 0, keys=LORA_KEYS):
    """LoRA pytree with mlx-lm's init: ``lora_a`` (in, r) uniform in
    ``+-1/sqrt(in)``, ``lora_b`` (r, out) zero, so the adapters start as a no-op.

    The matrices are float32 for any parameter dtype, as in mlx-lm. The tree is
    keyed by block index (as str) and projection name; ``keys`` restricts the
    adapted projections (default: all seven linear layers of a block).
    """
    rng = np.random.default_rng(seed)
    dims = _linear_dims(config)
    rank = config.lora_rank
    tree = {}
    for i in lora_layers(config):
        tree[str(i)] = {}
        for name in keys:
            fan_in, fan_out = dims[name]
            bound = 1.0 / math.sqrt(fan_in)
            lora_a = rng.uniform(-bound, bound, (fan_in, rank)).astype(np.float32)
            tree[str(i)][name] = {
                "lora_a": jnp.asarray(lora_a),
                "lora_b": jnp.zeros((rank, fan_out), jnp.float32),
            }
    return tree
