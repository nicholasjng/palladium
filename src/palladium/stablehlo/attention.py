"""Attention written out as softmax(q k') v, on the flash-attention kernel.

Plain JAX attention reaches the plugin as a product, a softmax over its last
axis and a second product. Scheduled as written, the scores take a buffer of
queries times keys elements, which the softmax then passes over three times.
`match` recognizes that shape of graph; `attention_source` is one kernel that
keeps the softmax in registers, so the scores never reach memory.

Only the forward form is recognized. In a gradient the probabilities have
other readers, so they must exist, and the program is scheduled as written.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

from palladium.device import prefers_simdgroup_matrix
from palladium.emit.tensorops import compile_kernel
from palladium.launch import launch_geometry
from palladium.stablehlo.graph import Constant, Node, Op
from palladium.trace import trace

# Key tiles of the kernel, and its smallest query tile.
TILE_K = 16
# Below this many scores a fused kernel saves nothing worth a pipeline.
MIN_SCORES = 1 << 16
# Wider heads do not fit the kernel's tiles in threadgroup memory.
MAX_HEAD = 128


@dataclasses.dataclass(frozen=True)
class Attention:
    """softmax(scale * q k') v over `heads` independent (queries, keys) pairs.

    `q`, `k` and `v` are the operands with their dimensions ordered
    (heads..., length, head_dim), which is also the layout of the result.
    """

    q: Node
    k: Node
    v: Node
    scale: float
    heads: int
    queries: int
    keys: int
    head_dim: int


def _through_broadcast(node: Node) -> Node:
    """`node` without the reshapes and broadcasts that keep a reduced dimension."""
    while isinstance(node, Op) and node.name in ("reshape", "broadcast_in_dim"):
        [node] = node.operands
    return node


def _relabelled(node: Node) -> Node | None:
    """The operand of a reshape, or of a broadcast that only adds unit
    dimensions, which keeps the last two dimensions: the same elements in
    the same order with their leading dimensions regrouped, as `jnp.matmul`
    wraps around a product of more than three dimensions."""
    if not isinstance(node, Op) or node.name not in ("reshape", "broadcast_in_dim"):
        return None
    [x] = node.operands
    kept = math.prod(x.shape) == math.prod(node.shape) and x.shape[-2:] == node.shape[-2:]
    dims = node.attributes.get("dims")
    return x if kept and (dims is None or list(dims) == sorted(dims)) else None


def _regrouped(node: Node, uses: Mapping[Node, int]) -> Node:
    """`node` without the relabellings that it alone reads."""
    while uses[node] == 1 and (inner := _relabelled(node)) is not None:
        node = inner
    return node


def _row_reduction(node: Node, of: Node, reducer: str) -> bool:
    """Whether `node` is `of` reduced by `reducer` over its last dimension."""
    node = _through_broadcast(node)
    if isinstance(node, Op) and node.name == "maximum" and reducer == "maximum":
        # jax.nn.softmax takes the maximum with an initial value of -inf.
        for kept, other in (node.operands, node.operands[::-1]):
            other = _through_broadcast(other)
            if isinstance(other, Constant) and other.splat == -math.inf:
                node = _through_broadcast(kept)
                break
    return (
        isinstance(node, Op)
        and node.name == "reduce"
        and node.operands[0] is of
        and node.attributes["reducer"] == reducer
        and list(node.attributes["dims"]) == [len(of.shape) - 1]
    )


def _ordered(x: Node, batch: tuple[int, ...], length: int, head: int) -> Node:
    """`x` with its dimensions ordered (batch..., length, head_dim)."""
    perm = (*batch, length, head)
    if perm == tuple(range(len(x.shape))):
        return x
    return Op("transpose", [x], x.dtype, tuple(x.shape[d] for d in perm), {"perm": perm})


def match(node: Op, uses: Mapping[Node, int]) -> Attention | None:
    """`node` as attention, if it is the product of a row softmax of a product
    with values, and nothing else reads the scores or the probabilities."""
    if node.name != "dot_general" or node.dtype != np.float32:
        return None
    p, v = node.operands
    a, rank = node.attributes, len(p.shape)
    batch = tuple(range(rank - 2))
    # The values contract with the softmax axis, over the same leading batch.
    if a["lhs_batch"] != batch or a["lhs_contract"] != (rank - 1,) or len(v.shape) != rank:
        return None
    if len(a["rhs_contract"]) != 1 or uses[p] != 1:
        return None
    p = _regrouped(p, uses)
    if not (isinstance(p, Op) and p.name == "divide" and uses[p] == 1):
        return None
    w, total = p.operands
    if not (isinstance(w, Op) and w.name == "exponential" and uses[w] == 2):
        return None
    if not _row_reduction(total, w, "add"):
        return None
    [shifted] = w.operands
    if not (isinstance(shifted, Op) and shifted.name == "subtract" and uses[shifted] == 1):
        return None
    scores, highest = shifted.operands
    if not (_row_reduction(highest, scores, "maximum") and uses[scores] == 2):
        return None
    # The scores: a product, possibly times a constant.
    scale, product = 1.0, scores
    if isinstance(scores, Op) and scores.name == "multiply":
        for product, factor in (scores.operands, scores.operands[::-1]):
            factor = _through_broadcast(factor)
            if isinstance(factor, Constant) and factor.splat is not None:
                scale = float(factor.splat)
                break
        else:
            return None
        if uses[product] != 1:
            return None
    if product is scores and (inner := _relabelled(scores)) is not None:
        # The subtraction and the maximum read the relabelled scores.
        product = inner
    product = _regrouped(product, uses) if uses[product] == 1 else product
    if not (isinstance(product, Op) and product.name == "dot_general"):
        return None
    q, k = product.operands
    b = product.attributes
    free = [
        tuple(d for d in range(len(x.shape)) if d not in b[f"{side}_batch"] + b[f"{side}_contract"])
        for side, x in (("lhs", q), ("rhs", k))
    ]
    if (
        any(x.dtype != np.float32 for x in (q, k, v))
        or len(b["lhs_contract"]) != 1
        or any(len(f) != 1 for f in free)
    ):
        return None
    [v_head] = (d for d in range(rank) if d not in a["rhs_batch"] + a["rhs_contract"])
    queries, keys, head_dim = scores.shape[-2], scores.shape[-1], v.shape[v_head]
    if (
        q.shape[b["lhs_contract"][0]] != head_dim
        or queries % 32
        or keys % TILE_K
        or head_dim % 16
        or head_dim > MAX_HEAD
        or queries * keys < MIN_SCORES
    ):
        return None
    return Attention(
        _ordered(q, b["lhs_batch"], free[0][0], b["lhs_contract"][0]),
        _ordered(k, b["rhs_batch"], free[1][0], b["rhs_contract"][0]),
        _ordered(v, a["rhs_batch"], a["rhs_contract"][0], v_head),
        scale,
        math.prod(scores.shape[:-2]),
        queries,
        keys,
        head_dim,
    )


def _kernel(q_ref, k_ref, v_ref, out_ref, *, tile_q, head_dim, scale):
    """One tile of queries against all keys, as an online softmax: the form
    the TensorOps recognizer lowers to one cooperative kernel."""
    q = q_ref[0, :, 0, :]

    def update(key_block, state):
        running_max, running_sum, running_output = state
        start = key_block * TILE_K
        k = k_ref[0, pl.dslice(start, TILE_K), 0, :]
        v = v_ref[0, pl.dslice(start, TILE_K), 0, :]
        scores = (q @ k.T) * jnp.asarray(scale, dtype=jnp.float32)
        active = jnp.ones((tile_q, TILE_K), dtype=jnp.bool_)
        new_max = jnp.maximum(running_max, jnp.max(scores, axis=1))
        old_scale = jnp.where(running_sum > 0.0, jnp.exp(running_max - new_max), 0.0)
        probabilities = jnp.where(active, jnp.exp(scores - new_max[:, None]), 0.0)
        new_sum = old_scale * running_sum + jnp.sum(probabilities, axis=1)
        return new_max, new_sum, old_scale[:, None] * running_output + probabilities @ v

    initial = (
        jnp.full((tile_q,), -jnp.inf, dtype=jnp.float32),
        jnp.zeros((tile_q,), dtype=jnp.float32),
        jnp.zeros((tile_q, head_dim), dtype=jnp.float32),
    )
    _, total, output = jax.lax.fori_loop(0, k_ref.shape[1] // TILE_K, update, initial)
    out_ref[0, :, 0, :] = output / total[:, None]


@functools.lru_cache(maxsize=64)
def attention_source(
    heads: int, queries: int, keys: int, head_dim: int, scale: float
) -> tuple[str, str, tuple[int, int, int], tuple[int, int, int]]:
    """The kernel's source, function name, thread grid and threadgroup.

    Its buffers are q (heads, queries, 1, head_dim), k and v (heads, keys, 1,
    head_dim) and the output, shaped as q: each head is one batch of the
    kernel, so operands ordered (heads..., length, head_dim) bind as stored.
    """
    tile_q = 64 if queries % 64 == 0 else 32
    block = pl.BlockSpec((1, tile_q, 1, head_dim), lambda b, h, i: (b, i, h, 0))
    whole = pl.BlockSpec((1, keys, 1, head_dim), lambda b, h, i: (b, 0, h, 0))
    f32 = jnp.float32
    call = pl.pallas_call(
        functools.partial(_kernel, tile_q=tile_q, head_dim=head_dim, scale=scale),
        grid=(heads, 1, queries // tile_q),
        in_specs=[block, whole, whole],
        out_specs=block,
        out_shape=jax.ShapeDtypeStruct((heads, queries, 1, head_dim), f32),
    )
    shapes = [(heads, n, 1, head_dim) for n in (queries, keys, keys)]
    spec = trace(call, *(jax.ShapeDtypeStruct(shape, f32) for shape in shapes))
    # Raises EmitError unless the kernel lowers cooperatively: emitted one
    # thread per program, its tiles would not fit a thread's stack.
    source, _ = compile_kernel(spec, "attention", prefer_simdgroup=prefers_simdgroup_matrix())
    grid, threadgroup = launch_geometry(spec, source, None)
    return source, "attention", grid, threadgroup or (0, 0, 0)
