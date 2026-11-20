"""Cooperative lowering for Pallas online-softmax attention."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext

import numpy as np
from jax.experimental import pallas as pl
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from palladium.device import device_limits
from palladium.emit.core import Cursor, CVal, literal
from palladium.errors import EmitError
from palladium.trace import KernelSpec

from ._shared import (
    SIMD_WIDTH,
    SIMDGROUPS,
    UNROLL,
    _index_map_is,
    _kernel_source,
    _producers,
    _shape,
    _tensor,
    _TensorOpsMatmul,
    _unsqueezed,
)
from .simdgroup import _MMA, emit_fragment_coordinates
from .softmax import (
    emit_online_softmax_fragments,
    emit_online_softmax_simd,
)

_BUILTINS = (
    "uint3 group [[threadgroup_position_in_grid]]",
    "uint tid [[thread_index_in_threadgroup]]",
    "uint lane [[thread_index_in_simdgroup]]",
    "uint sg [[simdgroup_index_in_threadgroup]]",
)
_PAD = 4  # floats per staged row; keeps fragment reads free of bank conflicts
_BUFFERS = (
    "device float* query [[buffer(0)]]",
    "device float* key [[buffer(1)]]",
    "device float* value [[buffer(2)]]",
    "device float* output [[buffer(3)]]",
)


@dataclasses.dataclass(frozen=True)
class _AttentionPlan:
    """The geometry of a recognized attention kernel; the algorithm itself is fixed."""

    query_length: int
    key_length: int
    heads: int
    dim: int
    tile_q: int
    tile_k: int
    causal: bool
    scale: str  # the constant the scores are scaled by, as MSL
    score_op: _TensorOpsMatmul
    value_op: _TensorOpsMatmul
    logsumexp: bool  # a second output stores each row's max + log(sum)


def lower_attention(
    spec: KernelSpec, kernel_name: str | None = None, *, prefer_simdgroup: bool = False
) -> tuple[str, int]:
    """Recognize a tiled online-softmax scan and emit cooperative attention.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``. The name must be a valid
        MSL identifier.
    prefer_simdgroup : bool, optional
        Compute both products with `simdgroup_matrix` and keep the softmax in
        registers when the tiles allow it (query tiles a multiple of 32, and staged
        K and V tiles within threadgroup memory); otherwise use `matmul2d`.

    Returns
    -------
    source : str
        Complete MSL using one threadgroup per Pallas program.
    threadgroup_bytes : int
        Explicitly declared shared storage per group.

    Raises
    ------
    EmitError
        The kernel dataflow, types, layouts, tiles, or device shared-memory budget do
        not match the supported cooperative form.

    Notes
    -----
    Requires rank-4 float32 Q/K/V, a standard 3D grid, one forward scan, and one non-aliased
    output, optionally followed by a (batch, heads, query length) output of each row's
    `running_max + log(running_sum)`, which only the `simdgroup_matrix` schedule stores.
    Tiles and head dimensions must be multiples of 16, or of 8 for the `simdgroup_matrix`
    schedule, and lengths must tile evenly. Noncausal query and key lengths may differ;
    causal lengths must match. Every equation of the kernel must belong to the
    online-softmax algorithm (`softmax.py`); see `_analyze_attention`.
    """
    scans = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "scan"]
    if len(scans) != 1:
        raise EmitError("tensorops attention requires one top-level scan region")
    (scan,) = scans
    body = scan.params["jaxpr"]
    if isinstance(body, ClosedJaxpr):
        body = body.jaxpr
    if not isinstance(body, Jaxpr):
        raise EmitError("tensorops attention scan has no jaxpr body")
    plan = _analyze_attention(spec, scan, body)
    staged_bytes = 2 * plan.tile_k * (plan.dim + _PAD) * 4
    if (
        prefer_simdgroup
        and plan.tile_q % (8 * SIMDGROUPS) == 0
        and staged_bytes <= device_limits().get("max_threadgroup_memory_length", 32768)
    ):
        return _emit_simdgroup(plan, spec, kernel_name)
    if plan.tile_q % 16 or plan.tile_k % 16 or plan.dim % 16:
        raise EmitError(
            "matmul2d attention tile and head dimensions must be multiples of 16; "
            'dot_general="simdgroup" takes multiples of 8 with query tiles of 32 rows'
        )
    return _emit(plan, spec, kernel_name)


class _Dataflow:
    """Producers of one jaxpr, and which equations the recognizer has matched.

    The emitters write the online-softmax algorithm themselves, so a kernel is
    that algorithm only if every equation has a role in it: `unmatched` lists
    the ones that have none.
    """

    def __init__(self, jaxpr: Jaxpr):
        self.eqns = jaxpr.eqns
        self.producers = _producers(jaxpr)
        self.matched: set[int] = set()

    def find(self, atom, *names: str) -> JaxprEqn | None:
        """The equation producing `atom` if it is one of the named primitives."""
        eqn = self.producers.get(atom) if isinstance(atom, Var) else None
        if eqn is None or eqn.primitive.name not in names:
            return None
        self.matched.add(id(eqn))
        return eqn

    def take(self, atom, name: str, role: str) -> JaxprEqn:
        """The `name` equation producing `atom`, or EmitError naming its role."""
        eqn = self.find(atom, name)
        if eqn is None:
            raise EmitError(f"tensorops attention expected `{name}` to produce the {role}")
        return eqn

    def split_by(self, atom, name: str, producer: str, role: str) -> tuple[JaxprEqn, object]:
        """For the commutative binary `name` producing `atom`: the `producer` equation
        behind one of its operands, and the other operand."""
        eqn = self.take(atom, name, role)
        found = [(inner, a) for a in eqn.invars if (inner := self.find(a, producer)) is not None]
        if len(eqn.invars) != 2 or len(found) != 1:
            raise EmitError(f"tensorops attention {role} must combine one `{producer}`")
        ((inner, operand),) = found
        return inner, next(a for a in eqn.invars if a is not operand)

    def split(self, atom, name: str, operand, role: str) -> tuple[JaxprEqn, object]:
        """The commutative binary `name` producing `atom`, and its operand other
        than `operand`."""
        eqn = self.take(atom, name, role)
        others = [value for value in eqn.invars if value is not operand]
        if len(eqn.invars) != 2 or len(others) != 1:
            raise EmitError(f"tensorops attention {role} has unexpected operands")
        return eqn, others[0]

    def constant(self, atom, value: float | bool) -> bool:
        """Whether `atom` is the scalar literal `value`, possibly converted or broadcast."""
        while (eqn := self.find(atom, "broadcast_in_dim", "convert_element_type")) is not None:
            if len(eqn.invars) != 1:
                return False
            atom = eqn.invars[0]
        return isinstance(atom, Literal) and np.ndim(atom.val) == 0 and atom.val == value

    def where(self, atom) -> tuple[object, object, object] | None:
        """`(predicate, if_true, if_false)` when `jnp.where` or `select_n` produces `atom`."""
        eqn = self.producers.get(atom) if isinstance(atom, Var) else None
        if eqn is None or len(eqn.invars) != 3:
            return None
        if eqn.primitive.name == "select_n":
            self.matched.add(id(eqn))
            predicate, if_false, if_true = eqn.invars
            return predicate, if_true, if_false
        if eqn.primitive.name != "jit":
            return None
        inner = eqn.params["jaxpr"]
        if isinstance(inner, ClosedJaxpr):
            if inner.consts:
                return None
            inner = inner.jaxpr
        flow = _Dataflow(inner)
        select = flow.find(inner.outvars[0], "select_n")
        if select is None or len(select.invars) != 3 or len(inner.outvars) != 1:
            return None
        predicate, if_false, if_true = select.invars

        def argument(value):  # jnp.where converts and broadcasts its branches
            while (
                cast := flow.find(value, "broadcast_in_dim", "convert_element_type")
            ) is not None:
                if len(cast.invars) != 1:
                    return None
                value = cast.invars[0]
            return value

        if (
            predicate is not inner.invars[0]
            or argument(if_true) is not inner.invars[1]
            or argument(if_false) is not inner.invars[2]
            or flow.unmatched()
        ):
            return None
        self.matched.add(id(eqn))
        return eqn.invars[0], eqn.invars[1], eqn.invars[2]

    def unmatched(self) -> list[JaxprEqn]:
        return [eqn for eqn in self.eqns if id(eqn) not in self.matched]


def _ref_indices(eqn: JaxprEqn) -> tuple:
    """The indices of a `get` or `swap` with one indexer; () for a full-block access."""
    transforms = eqn.params["tree"].unflatten(eqn.invars[1 if eqn.primitive.name == "get" else 2 :])
    if not transforms:
        return ()
    if len(transforms) != 1 or not hasattr(transforms[0], "indices"):
        raise EmitError("tensorops attention indexes refs with scalars and slices")
    return tuple(transforms[0].indices)


def _is_zero(index) -> bool:
    if isinstance(index, Literal):
        index = index.val
    return isinstance(index, int | np.integer | np.ndarray) and np.ndim(index) == 0 and index == 0


def _is_slice(index, start, size: int) -> bool:
    """Whether `index` is the unit-stride slice of `size` from `start`: a jaxpr atom,
    matched by identity, or None for zero."""
    if not isinstance(index, pl.Slice) or index.size != size or index.stride != 1:
        return False
    return _is_zero(index.start) if start is None else index.start is start


def _row_broadcast(flow: _Dataflow, atom, rows: int, role: str):
    """The `(rows,)` operand of the `[:, None]` broadcast producing `atom`."""
    eqn = flow.take(atom, "broadcast_in_dim", role)
    if (
        tuple(eqn.params["shape"]) != (rows, 1)
        or tuple(eqn.params["broadcast_dimensions"]) != (0,)
        or len(eqn.invars) != 1
    ):
        raise EmitError(f"tensorops attention {role} must broadcast one value per query row")
    return eqn.invars[0]


def _analyze_attention(spec: KernelSpec, scan: JaxprEqn, body: Jaxpr) -> _AttentionPlan:
    """Match the whole kernel against online-softmax attention.

    The emitters reproduce the algorithm rather than translate equations, so
    this accepts a kernel only when every equation of the scan body and of
    the program around it is one of the algorithm's, wired as it requires.
    Anything else raises EmitError, and the "auto" policy then emits the
    kernel as written.
    """
    if scan.params.get("reverse") or scan.params.get("unroll") != 1:
        raise EmitError("tensorops attention requires a forward, single-step scan")
    if len(spec.inputs) != 3 or len(spec.outputs) not in (1, 2) or spec.scratch or spec.aliases:
        raise EmitError(
            "tensorops attention requires Q, K, V, an output and optionally its logsumexp"
        )
    if len(spec.grid) != 3:
        raise EmitError("tensorops attention requires a complete three-dimensional Pallas grid")

    query, key, value = spec.inputs
    output = spec.outputs[0]
    if any(
        info.dtype != np.float32 or len(info.array_shape) != 4 for info in (*spec.inputs, output)
    ):
        raise EmitError("tensorops attention requires rank-4 float32 buffers")
    batch, query_length, heads, dim = query.array_shape
    key_batch, key_length, key_heads, key_dim = key.array_shape
    if (
        output.array_shape != query.array_shape
        or value.array_shape != key.array_shape
        or (key_batch, key_heads, key_dim) != (batch, heads, dim)
    ):
        raise EmitError("tensorops attention Q, K, V, and output shapes are incompatible")
    if value.full_block_shape != key.full_block_shape:
        raise EmitError("tensorops attention requires explicit query and key/value block shapes")
    q_block, k_block = _unsqueezed(query.full_block_shape), _unsqueezed(key.full_block_shape)
    if q_block[0] != 1 or q_block[2:] != (1, dim) or k_block != (1, key_length, 1, dim):
        raise EmitError("tensorops attention requires (1, BQ, 1, D) and full-sequence K/V blocks")
    tile_q = q_block[1]
    scan_length = int(scan.params["length"])
    if (
        key_length % scan_length
        or query_length % tile_q
        or output.full_block_shape != query.full_block_shape
        or spec.grid != (batch, heads, query_length // tile_q)
        or not (
            _index_map_is(query, (0, 2, 1, None))
            and _index_map_is(key, (0, None, 1, None))
            and _index_map_is(value, (0, None, 1, None))
            and _index_map_is(output, (0, 2, 1, None))
        )
    ):
        raise EmitError("tensorops attention requires standard full-sequence block mappings")
    tile_k = key_length // scan_length
    if tile_q % 8 or tile_k % 8 or dim % 8:
        raise EmitError("TensorOps attention tile and head dimensions must be multiples of 8")

    # The loop carries a counter, each row's running maximum and sum, and the
    # weighted values; everything else the body reads is loop-invariant.
    if (
        len(body.invars) != len(scan.invars)
        or len(body.outvars) != 4
        or len(scan.outvars) != 4
        or len(body.invars) < 4
    ):
        raise EmitError("tensorops attention scan must carry a counter, maxima, sums and values")
    counter, max_carry, sum_carry, value_carry = body.invars[-4:]
    if tuple(_shape(atom) for atom in body.invars[-4:]) != (
        (),
        (tile_q,),
        (tile_q,),
        (tile_q, dim),
    ):
        raise EmitError("tensorops attention scan carries do not match the tile geometry")
    arguments = dict(zip(body.invars, scan.invars, strict=True))

    def operand(atom):  # what the scan passes for a body input; None for anything else
        return arguments.get(atom) if isinstance(atom, Var) else None

    q_ref, k_ref, v_ref, out_ref = spec.jaxpr.invars[:4]
    flow, outer = _Dataflow(body), _Dataflow(spec.jaxpr)
    outer.matched.add(id(scan))
    next_counter, new_max, new_sum, new_value = body.outvars

    _, step = flow.split(next_counter, "add", counter, "loop counter")
    if not (isinstance(step, Literal) and int(step.val) == 1):
        raise EmitError("tensorops attention scan must advance one key tile per step")

    # Weighted values: old_scale[:, None] * running_output + probabilities @ v.
    value_dot, rescaled = flow.split_by(new_value, "add", "dot_general", "weighted-value update")
    _, row_scale = flow.split(rescaled, "mul", value_carry, "accumulator rescale")
    old_scale = _row_broadcast(flow, row_scale, tile_q, "accumulator rescale")
    probabilities, value_tile = value_dot.invars

    # Running maximum over the scaled, optionally masked scores.
    _, block_max = flow.split(new_max, "max", max_carry, "running maximum update")
    score_max = flow.take(block_max, "reduce_max", "row maximum")
    scores = score_max.invars[0]
    masked = flow.where(scores)
    mask = None
    if masked is not None:
        mask, scaled, fill = masked
        if not flow.constant(fill, -np.inf):
            raise EmitError(
                "tensorops causal score maximum must mask future scores to negative infinity"
            )
    else:
        scaled = scores
    score_dot, scale = flow.split_by(scaled, "mul", "dot_general", "score scale")
    scale_literal = scale if isinstance(scale, Literal) else operand(scale)
    if not isinstance(scale_literal, Literal) or np.ndim(scale_literal.val) != 0:
        raise EmitError("tensorops attention requires a constant softmax scale")
    expected = (
        ((tile_q, dim), (dim, tile_k), (tile_q, tile_k)),
        ((tile_q, tile_k), (tile_k, dim), (tile_q, dim)),
    )
    for dot, shapes in zip((score_dot, value_dot), expected, strict=True):
        if tuple(_shape(atom) for atom in (*dot.invars, *dot.outvars)) != shapes:
            raise EmitError("tensorops attention dot shapes do not match the tile geometry")

    # Both products read the same key tile of K and V; Q is this program's block.
    query_tile, keys_transposed = score_dot.invars
    transpose = flow.take(keys_transposed, "transpose", "key tile of the score product")
    key_start = None
    for tile, ref, role in ((transpose.invars[0], k_ref, "key"), (value_tile, v_ref, "value")):
        read = flow.take(tile, "get", f"{role} tile")
        indices = _ref_indices(read)
        if operand(read.invars[0]) is not ref or len(indices) != 4:
            raise EmitError(f"tensorops attention must read {role} tiles from its {role} operand")
        if key_start is None and isinstance(indices[1], pl.Slice):
            key_start = indices[1].start
        if not (
            isinstance(key_start, Var)
            and _is_zero(indices[0])
            and _is_slice(indices[1], key_start, tile_k)
            and _is_zero(indices[2])
            and _is_slice(indices[3], None, dim)
        ):
            raise EmitError(
                "tensorops attention must read the key and value tile of the current scan step"
            )
    _, tile_size = flow.split(key_start, "mul", counter, "key tile offset")
    if not (isinstance(tile_size, Literal) and int(tile_size.val) == tile_k):
        raise EmitError("tensorops attention key tiles must follow the scan's key-tile index")
    query_read = outer.take(operand(query_tile), "get", "query tile")
    indices = _ref_indices(query_read)
    if query_read.invars[0] is not q_ref or not (
        len(indices) == 4
        and _is_zero(indices[0])
        and _is_slice(indices[1], None, tile_q)
        and _is_zero(indices[2])
        and _is_slice(indices[3], None, dim)
    ):
        raise EmitError("tensorops attention must take its queries from the whole Q block")

    # old_scale = where(running_sum > 0, exp(running_max - new_max), 0).
    rescale = flow.where(old_scale)
    if rescale is None or not flow.constant(rescale[2], 0.0):
        raise EmitError("tensorops attention must rescale only rows with a nonempty running sum")
    nonempty = flow.take(rescale[0], "gt", "nonempty-row predicate")
    decay = flow.take(rescale[1], "exp", "running-state rescale")
    drop = flow.take(decay.invars[0], "sub", "running-state rescale")
    if (
        nonempty.invars[0] is not sum_carry
        or not flow.constant(nonempty.invars[1], 0.0)
        or drop.invars[0] is not max_carry
        or drop.invars[1] is not new_max
    ):
        raise EmitError(
            "tensorops attention must rescale running state by exp(running_max - new_max)"
        )

    # probabilities = exp(scores - new_max[:, None]), optionally zeroed off the mask.
    selected = flow.where(probabilities)
    if selected is not None:
        active, exponential, fill = selected
        if not flow.constant(fill, 0.0) or not (
            flow.constant(active, True) if mask is None else active is mask
        ):
            raise EmitError("tensorops softmax probabilities must be zero exactly off the mask")
    else:
        exponential = probabilities
    probability_exp = flow.take(exponential, "exp", "probability exponential")
    probability_center = flow.take(probability_exp.invars[0], "sub", "probability centering")
    if (
        probability_center.invars[0] is not scores
        or flow.find(probability_center.invars[1], "broadcast_in_dim") is None
        or _row_broadcast(flow, probability_center.invars[1], tile_q, "probability centering")
        is not new_max
    ):
        raise EmitError(
            "tensorops softmax probabilities must compute scores minus the running maximum"
        )

    # new_sum = old_scale * running_sum + sum(probabilities).
    probability_sum, scaled_sum = flow.split_by(new_sum, "add", "reduce_sum", "running sum update")
    _, sum_factor = flow.split(scaled_sum, "mul", sum_carry, "running sum rescale")
    if any(tuple(eqn.params["axes"]) != (1,) for eqn in (score_max, probability_sum)):
        raise EmitError("tensorops attention must reduce scores over the keys of each row")
    if probability_sum.invars[0] is not probabilities or sum_factor is not old_scale:
        raise EmitError("tensorops attention must sum the probabilities it weights the values with")

    if mask is not None:
        if query_length != key_length:
            raise EmitError("tensorops attention supports at most one square causal comparison")
        comparison = flow.take(mask, "le", "causal mask")
        _match_causal_positions(flow, outer, operand, comparison, key_start, tile_q, tile_k)

    # Around the scan: empty initial state, the normalized output, and optionally
    # each row's logsumexp.
    start, *initial = scan.invars[-4:]
    if not (isinstance(start, Literal) and int(start.val) == 0) or not (
        outer.constant(initial[0], -np.inf)
        and outer.constant(initial[1], 0.0)
        and outer.constant(initial[2], 0.0)
    ):
        raise EmitError("tensorops attention must start at key tile zero with empty running state")
    _, final_max, final_sum, final_value = scan.outvars
    stores = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "swap"]
    store = next((eqn for eqn in stores if eqn.invars[0] is out_ref), None)
    if store is None or len(stores) != len(spec.outputs):
        raise EmitError("tensorops attention stores each output once")
    outer.matched.add(id(store))
    indices = _ref_indices(store)
    final_div = outer.take(store.invars[1], "div", "final attention normalization")
    if (
        final_div.invars[0] is not final_value
        or _row_broadcast(outer, final_div.invars[1], tile_q, "final attention normalization")
        is not final_sum
        or not (
            len(indices) == 4
            and _is_zero(indices[0])
            and _is_slice(indices[1], None, tile_q)
            and _is_zero(indices[2])
            and _is_slice(indices[3], None, dim)
        )
    ):
        raise EmitError("tensorops attention must store the weighted values over the running sum")
    logsumexp = len(spec.outputs) == 2
    if logsumexp:
        _match_logsumexp(spec, outer, stores, final_max, final_sum, tile_q)

    stray = [*flow.unmatched(), *outer.unmatched()]
    if stray:
        names = ", ".join(sorted({eqn.primitive.name for eqn in stray}))
        raise EmitError(
            f"tensorops attention kernel has operations outside online-softmax attention: {names}"
        )

    producers = flow.producers
    return _AttentionPlan(
        query_length=query_length,
        key_length=key_length,
        heads=heads,
        dim=dim,
        tile_q=tile_q,
        tile_k=tile_k,
        causal=mask is not None,
        scale=literal(scale_literal).expr,
        score_op=_TensorOpsMatmul.from_eqn(score_dot, producers, name="score_op", accumulate=False),
        value_op=_TensorOpsMatmul.from_eqn(value_dot, producers, name="value_op", accumulate=True),
        logsumexp=logsumexp,
    )


def _match_logsumexp(
    spec: KernelSpec, outer: _Dataflow, stores: list[JaxprEqn], final_max, final_sum, tile_q: int
) -> None:
    """Require the second output to store `running_max + log(running_sum)` per query row."""
    query, info = spec.inputs[0], spec.outputs[1]
    batch, query_length, heads, _ = query.array_shape
    if (
        info.dtype != np.float32
        or tuple(info.array_shape) != (batch, heads, query_length)
        or _unsqueezed(info.full_block_shape) != (1, 1, tile_q)
        or not _index_map_is(info, (0, 1, 2))
    ):
        raise EmitError(
            "tensorops attention stores the logsumexp as (batch, heads, query length) "
            "in (1, 1, BQ) blocks"
        )
    ref = spec.jaxpr.invars[len(spec.inputs) + 1]
    store = next((eqn for eqn in stores if eqn.invars[0] is ref), None)
    total = None if store is None else outer.find(store.invars[1], "add")
    if store is not None and total is not None and any(a is final_max for a in total.invars):
        outer.matched.add(id(store))
        indices = _ref_indices(store)
        log = outer.find(next(a for a in total.invars if a is not final_max), "log")
        if (
            log is not None
            and log.invars[0] is final_sum
            and len(indices) == 3
            and _is_zero(indices[0])
            and _is_zero(indices[1])
            and _is_slice(indices[2], None, tile_q)
        ):
            return
    raise EmitError("tensorops attention's second output must store running_max + log(running_sum)")


def _match_causal_positions(
    flow: _Dataflow,
    outer: _Dataflow,
    operand,
    mask: JaxprEqn,
    key_start: Var,
    tile_q: int,
    tile_k: int,
) -> None:
    """Require the mask to compare absolute key and query positions for this grid tile:
    `key_start + arange(tile_k) <= program_id(2) * tile_q + arange(tile_q)`."""
    key_position = flow.find(mask.invars[0], "broadcast_in_dim")
    if (
        _shape(mask.outvars[0]) != (tile_q, tile_k)
        or key_position is None
        or tuple(key_position.params["shape"]) != (1, tile_k)
        or tuple(key_position.params["broadcast_dimensions"]) != (1,)
    ):
        raise EmitError(
            "tensorops causal mask must broadcast absolute positions over rows and columns"
        )
    query_positions = operand(_row_broadcast(flow, mask.invars[1], tile_q, "causal mask"))

    def tile_positions(dataflow: _Dataflow, atom, tile: int, role: str):
        """The offset added to `arange(tile)` to produce `atom`."""
        total = dataflow.take(atom, "add", role)
        iota = next(
            (eqn for a in total.invars if (eqn := dataflow.find(a, "iota")) is not None), None
        )
        if iota is None or tuple(iota.params["shape"]) != (tile,) or len(total.invars) != 2:
            raise EmitError(f"tensorops {role} need an iota matching the tile")
        return next(a for a in total.invars if a is not iota.outvars[0])

    key_offset = tile_positions(flow, key_position.invars[0], tile_k, "causal key coordinates")
    converted = flow.find(key_offset, "convert_element_type")
    if (key_offset if converted is None else converted.invars[0]) is not key_start:
        raise EmitError("tensorops causal key offset must follow the scan's key-tile index")
    query_offset = tile_positions(outer, query_positions, tile_q, "causal query coordinates")
    converted = outer.find(query_offset, "convert_element_type")
    scaled = outer.find(query_offset if converted is None else converted.invars[0], "mul")
    program_id = None
    if scaled is not None and len(scaled.invars) == 2:
        program_id = next(
            (eqn for a in scaled.invars if (eqn := outer.find(a, "program_id")) is not None), None
        )
    if (
        scaled is None
        or program_id is None
        or program_id.params.get("axis") != 2
        or not any(isinstance(a, Literal) and int(a.val) == tile_q for a in scaled.invars)
    ):
        raise EmitError("tensorops causal query positions must use the query-tile grid axis")


def _emit(plan: _AttentionPlan, spec: KernelSpec, kernel_name: str | None):
    """Emit attention with `matmul2d` products: scores in threadgroup memory, row
    state shared out across SIMD groups, and the weighted values in the value
    product's cooperative tensor."""
    query_length, key_length = plan.query_length, plan.key_length
    heads, dim = plan.heads, plan.dim
    tile_q, tile_k, causal = plan.tile_q, plan.tile_k, plan.causal
    score_op, value_op = plan.score_op, plan.value_op
    if plan.logsumexp:
        raise EmitError(
            'the attention logsumexp output needs dot_general="simdgroup" or "auto" on '
            "Apple GPU families 7-9, with query tiles of 32 rows"
        )
    name = kernel_name or spec.name
    lanes = SIMD_WIDTH
    if tile_q % SIMDGROUPS:
        raise EmitError(f"attention query tiles must be a multiple of {SIMDGROUPS} rows")
    cursor = Cursor()
    cursor.emit(f"constexpr int BQ = {tile_q};")
    cursor.emit(f"constexpr int BK = {tile_k};")
    cursor.emit(f"constexpr int D = {dim};")
    cursor.emit(
        "const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;"
    )
    cursor.emit("const uint batch = group.x;")
    cursor.emit("const uint head = group.y;")
    cursor.emit("const uint q_start = group.z * BQ;")
    cursor.emit(f"const uint q_base = (batch * {query_length} * {heads} + head) * D;")
    cursor.emit(f"const uint kv_base = (batch * {key_length} * {heads} + head) * D;")

    scores = cursor.allocate("float", (tile_q, tile_k), name="scores", space="threadgroup")
    row_max = cursor.allocate("float", (tile_q,), name="row_max", space="threadgroup")
    row_sum = cursor.allocate("float", (tile_q,), name="row_sum", space="threadgroup")
    row_scale = cursor.allocate("float", (tile_q,), name="row_scale", space="threadgroup")
    out_tile = CVal(
        f"(output + q_base + q_start * {heads * dim})",
        (tile_q * heads * dim,),
        "float",
        space="device",
    )

    with cursor.strided_loop("tid", "BQ", "THREADS", name="row"):
        cursor.emit(f"{row_max.at('row')} = -INFINITY;")
        cursor.emit(f"{row_sum.at('row')} = 0.0f;")
    score_op.emit_declaration(cursor)
    value_op.emit_declaration(cursor)
    # Q is read from device on every key step: staging it in threadgroup
    # memory costs more occupancy than the reads cost bandwidth.
    stride = heads * dim

    def rows(name: str, pointer: str, count: str) -> str:
        """A tensor of `count` rows of D features of one head, from `pointer`."""
        return _tensor(cursor, name, "device float", pointer, ("D", count), (1, stride))

    q_tile = rows("q_tile", f"query + q_base + q_start * {stride}", "BQ")
    score_tile = _tensor(
        cursor, "score_tile", "threadgroup float", scores.expr, ("BK", "BQ"), (1, "BK")
    )
    # A view at the head base fixes the value operand's type for the
    # destination; each key step builds its own view at k_start.
    value_view = rows("value_view", "value + kv_base", "BK")
    # The accumulator is the value matmul's cooperative tensor, held in
    # per-thread registers for the whole key loop. MPP reports each element's
    # coordinate as (column, row).
    acc = value_op.emit_cooperative_destination(
        cursor, score_tile, value_view, name="acc", element_type="float"
    )
    with cursor.loop(f"{acc}.get_capacity()", "_i") as i:
        cursor.emit(f"{acc}[{i}] = 0.0f;")
    cursor.barrier()

    key_loop_stop = "(((q_start + BQ + BK - 1) / BK) * BK)" if causal else str(key_length)
    with cursor.strided_loop("0", key_loop_stop, "BK", name="k_start"):
        k_tile = rows("k_tile", f"key + kv_base + k_start * {stride}", "BK")
        cursor.emit(f"{score_op.name}.run({q_tile}, {k_tile}, {score_tile});")
        cursor.barrier()

        emit_online_softmax_simd(
            cursor,
            scores,
            row_max,
            row_sum,
            row_scale,
            plan.scale,
            rows=tile_q,
            columns=tile_k,
            lanes=lanes,
            simdgroups=SIMDGROUPS,
            causal_offsets=("k_start", "q_start") if causal else None,
        )
        cursor.barrier()

        with cursor.loop(f"{acc}.get_capacity()", "_i") as i:
            cursor.emit(f"const int acc_row = {acc}.get_multidimensional_index({i})[1];")
            cursor.emit(f"{acc}[{i}] = ({row_scale.at('acc_row')} * {acc}[{i}]);")
        value_tile = rows("value_tile", f"value + kv_base + k_start * {stride}", "BK")
        cursor.emit(f"{value_op.name}.run({score_tile}, {value_tile}, {acc});")
        cursor.barrier()

    # Small tiles leave some slots unowned; they report another lane's
    # coordinates, so storing them would overwrite that lane's output. The
    # constant first test folds the guard away when slots cover the tile.
    fully_owned = f"{acc}.get_capacity() * {SIMDGROUPS * lanes} == {tile_q * dim}"
    with (
        cursor.loop(f"{acc}.get_capacity()", "_i") as i,
        cursor.block(f"if ({fully_owned} || {acc}.is_valid_element({i}))"),
    ):
        cursor.emit(f"const auto acc_index = {acc}.get_multidimensional_index({i});")
        cursor.emit("const int row = acc_index[1];")
        cursor.emit("const int column = acc_index[0];")
        normalized = f"({acc}[{i}] / {row_sum.at('row')})"
        cursor.emit(f"{out_tile.at(f'row * {heads * dim} + column')} = {normalized};")

    params = (*_BUFFERS, *_BUILTINS, "uint3 threads_per_group [[threads_per_threadgroup]]")
    source = _kernel_source(name, params, cursor)
    return source, cursor.threadgroup_bytes


def _emit_simdgroup(plan: _AttentionPlan, spec: KernelSpec, kernel_name: str | None):
    """Emit attention with `simdgroup_matrix` products and the softmax in registers.

    After MLX's steel attention. SIMD group `sg` owns `tile_q / 4` consecutive
    query rows, as 8x8 fragment rows. Per key step, all threads stage the K and
    V tiles in threadgroup memory; each SIMD group then forms its scores, updates
    the softmax and accumulates the output in its own registers, so a key step
    needs only the two barriers around staging.
    """
    tile_q, tile_k, dim, heads = plan.tile_q, plan.tile_k, plan.dim, plan.heads
    rows = tile_q // SIMDGROUPS
    fragment_rows, key_fragments, dim_fragments = rows // 8, tile_k // 8, dim // 8
    stride, ld = heads * dim, dim + _PAD
    u = UNROLL
    cursor = Cursor()
    cursor.require("simdgroup_mma", _MMA)
    keys = cursor.allocate("float", (tile_k * ld,), name="_K", space="threadgroup")
    values = cursor.allocate("float", (tile_k * ld,), name="_V", space="threadgroup")
    cursor.emit(f"const uint q_start = group.z * {tile_q};")
    cursor.emit(f"const uint q_base = (group.x * {plan.query_length} * {heads} + group.y) * {dim};")
    cursor.emit(f"const uint kv_base = (group.x * {plan.key_length} * {heads} + group.y) * {dim};")
    emit_fragment_coordinates(cursor, "lane")
    cursor.emit("// This lane's query row in fragment row i is _row0 + 8 * i.")
    cursor.emit(f"const uint _row0 = q_start + sg * {rows} + _fm;")
    cursor.emit(f"simdgroup_frag _q[{fragment_rows}][{dim_fragments}];")
    cursor.emit(f"simdgroup_frag _o[{fragment_rows}][{dim_fragments}];")
    cursor.emit(f"float _m[{fragment_rows}], _l[{fragment_rows}];")
    with cursor.block(f"{u} for (short i = 0; i < {fragment_rows}; ++i)"):
        cursor.emit("_m[i] = -INFINITY;")
        cursor.emit("_l[i] = 0.0f;")
        cursor.emit(f"device const float* _qr = query + q_base + (_row0 + 8 * i) * {stride} + _fn;")
        with cursor.block(f"{u} for (short d = 0; d < {dim_fragments}; ++d)"):
            cursor.emit("_q[i][d] = simdgroup_frag(_qr[8 * d], _qr[8 * d + 1]);")
            cursor.emit("_o[i][d] = simdgroup_frag(0);")

    vectors = tile_k * dim // 4
    threads = SIMDGROUPS * SIMD_WIDTH
    copies = -(-vectors // threads)
    stop = (
        f"(((q_start + {tile_q} + {tile_k} - 1) / {tile_k}) * {tile_k})"
        if plan.causal
        else str(plan.key_length)
    )
    with cursor.block(f"for (uint k_start = 0; k_start < {stop}; k_start += {tile_k})"):
        cursor.barrier()
        with cursor.block(f"{u} for (short _c = 0; _c < {copies}; ++_c)"):
            cursor.emit(f"const short _e = tid + _c * {threads};")
            guard = cursor.block(f"if (_e < {vectors})") if vectors % threads else nullcontext()
            with guard:
                cursor.emit(f"const short _r = _e / {dim // 4}, _col = (_e % {dim // 4}) * 4;")
                cursor.emit(f"const uint _src = kv_base + (k_start + _r) * {stride} + _col;")
                for staged, source in ((keys, "key"), (values, "value")):
                    cursor.emit(
                        f"*(threadgroup float4*)({staged.expr} + _r * {ld} + _col) = "
                        f"*(device const float4*)({source} + _src);"
                    )
        cursor.barrier()
        cursor.emit(f"simdgroup_frag _s[{fragment_rows}][{key_fragments}];")
        with (
            cursor.block(f"{u} for (short i = 0; i < {fragment_rows}; ++i)"),
            cursor.block(f"{u} for (short j = 0; j < {key_fragments}; ++j)"),
        ):
            cursor.emit("_s[i][j] = simdgroup_frag(0);")
        cursor.emit("// Scores: K is staged by key, so the K^T fragment reads down columns.")
        with (
            cursor.block(f"{u} for (short d = 0; d < {dim_fragments}; ++d)"),
            cursor.block(f"{u} for (short j = 0; j < {key_fragments}; ++j)"),
        ):
            column = f"{keys.expr}[(8 * j + _fn) * {ld} + 8 * d + _fm]"
            cursor.emit(
                f"simdgroup_frag _b = simdgroup_frag({column}, "
                f"{keys.expr}[(8 * j + _fn + 1) * {ld} + 8 * d + _fm]);"
            )
            with cursor.block(f"{u} for (short i = 0; i < {fragment_rows}; ++i)"):
                cursor.emit("simdgroup_mma(_s[i][j], _q[i][d], _b);")
        emit_online_softmax_fragments(
            cursor,
            plan.scale,
            columns=tile_k,
            width=dim,
            fragment_rows=fragment_rows,
            causal=(
                f"k_start + {tile_k - 1} > q_start",
                "k_start + 8 * j + _fn + c",
                "_row0 + 8 * i",
            )
            if plan.causal
            else None,
        )
        cursor.emit("// Probabilities are already in the A-operand layout of the next product.")
        with (
            cursor.block(f"{u} for (short j = 0; j < {key_fragments}; ++j)"),
            cursor.block(f"{u} for (short d = 0; d < {dim_fragments}; ++d)"),
        ):
            row = f"{values.expr}[(8 * j + _fm) * {ld} + 8 * d + _fn"
            cursor.emit(f"simdgroup_frag _b = simdgroup_frag({row}], {row} + 1]);")
            with cursor.block(f"{u} for (short i = 0; i < {fragment_rows}; ++i)"):
                cursor.emit("simdgroup_mma(_o[i][d], _s[i][j], _b);")

    with cursor.block(f"{u} for (short i = 0; i < {fragment_rows}; ++i)"):
        cursor.emit(f"device float* _out = output + q_base + (_row0 + 8 * i) * {stride} + _fn;")
        with (
            cursor.block(f"{u} for (short d = 0; d < {dim_fragments}; ++d)"),
            cursor.block(f"{u} for (short c = 0; c < 2; ++c)"),
        ):
            cursor.emit("_out[8 * d + c] = (_o[i][d][c] / _l[i]);")
        if plan.logsumexp:  # the four lanes of a row hold the same state; one writes it
            row = f"(group.x * {heads} + group.y) * {plan.query_length} + _row0 + 8 * i"
            with cursor.block("if (_fn == 0)"):
                cursor.emit(f"logsumexp[{row}] = _m[i] + precise::log(_l[i]);")

    params = (
        *_BUFFERS,
        *(("device float* logsumexp [[buffer(4)]]",) if plan.logsumexp else ()),
        *_BUILTINS,
    )
    source = _kernel_source(kernel_name or spec.name, params, cursor)
    return source, cursor.threadgroup_bytes
