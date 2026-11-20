"""Cooperative lowering for a tiled matrix product.

`_recognize` matches the kernel, `_placement` decides how a `matmul2d` product
reaches the output, and `_lower_matmul2d` and `_lower_simdgroup` emit the two
kernels.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable

from jax.extend.core import JaxprEqn, Literal, Var

from palladium.emit.addressing import block_offset
from palladium.emit.core import CTYPES, Cursor, CVal, Environment, msl_type, shaped
from palladium.emit.numeric import ELEMENTWISE, HELPERS, format_scalar, typed_expression
from palladium.errors import EmitError
from palladium.trace import KernelSpec

from . import simdgroup
from ._shared import (
    _index_map_is,
    _kernel_source,
    _producers,
    _shape,
    _tensor,
    _TensorOpsMatmul,
)

_K_TILE = 128


@dataclasses.dataclass(frozen=True)
class _Product:
    """A recognized tiled product `(m, k) @ (k, n)` in `tm x tn` output tiles.

    Attributes
    ----------
    dot : JaxprEqn
        The product.
    gets : list of JaxprEqn
        Every ref read: the two operands, then tiles and biases the epilogue reads.
    epilogue : list of JaxprEqn
        Elementwise equations between the product and the store, in dependency order.
    store_value : Var
        What the kernel stores: the product, or the epilogue's result.
    rank : int
        2 for matrices, 3 with a leading batch axis on the grid's first axis.
    """

    spec: KernelSpec
    dot: JaxprEqn
    gets: list[JaxprEqn]
    epilogue: list[JaxprEqn]
    store_value: Var
    rank: int
    m: int
    n: int
    k: int
    tm: int
    tn: int
    transpose_lhs: bool
    transpose_rhs: bool

    @property
    def ctype(self) -> str:
        """The element type of every operand and of the output."""
        return msl_type(self.spec.outputs[0].dtype)

    @property
    def edge_tiles(self) -> bool:
        return self.m % self.tm != 0 or self.n % self.tn != 0

    @property
    def epilogue_reads_only_product(self) -> bool:
        """Whether the epilogue needs nothing but the product and literals."""
        known = {self.dot.outvars[0]}.union(*(eqn.outvars for eqn in self.epilogue))
        return all(
            isinstance(atom, Literal) or atom in known
            for eqn in self.epilogue
            for atom in eqn.invars
        )

    def parameters(self, *builtins: str) -> tuple[str, ...]:
        """The kernel's buffers, inputs then output, its grid position and `builtins`."""
        buffers = (
            f"device {msl_type(info.dtype)}* arg{index} [[buffer({index})]]"
            for index, info in enumerate((*self.spec.inputs, self.spec.outputs[0]))
        )
        return (*buffers, "uint3 _pid [[threadgroup_position_in_grid]]", *builtins)

    def narrowed(self, element: str) -> str:
        """A float32 accumulator element in the output type. Explicit, since Metal
        does not convert float to bfloat implicitly."""
        return element if self.ctype == "float" else f"{self.ctype}({element})"


class _Store(enum.Enum):
    """How a `matmul2d` product leaves its cooperative tensor."""

    #: Straight into the output buffer.
    DEVICE = enum.auto()
    #: Full tiles as DEVICE; edge tiles through threadgroup memory, masked.
    MASKED = enum.auto()
    #: Every tile through threadgroup memory, finished per element.
    STAGED = enum.auto()


def _placement(product: _Product) -> tuple[_Store, bool]:
    """Decide how a `matmul2d` product reaches the output.

    Returns
    -------
    store : _Store
        Where the cooperative tensor is stored.
    cooperative_epilogue : bool
        Whether the epilogue runs on the cooperative tensor before the store.

    Notes
    -----
    The product accumulates in float32, and the cooperative tensor can neither
    narrow on `store` nor read other tiles. So a half or bfloat output, or an
    epilogue that reads another operand, is STAGED: a per-element loop narrows
    the product, applies the epilogue and masks edge tiles. Otherwise the
    epilogue, if any, runs on the cooperative tensor. Either way it runs once.
    """
    if product.ctype != "float" or (product.epilogue and not product.epilogue_reads_only_product):
        return _Store.STAGED, False
    return (_Store.MASKED if product.edge_tiles else _Store.DEVICE), bool(product.epilogue)


def lower_matmul(
    spec: KernelSpec, kernel_name: str | None = None, *, prefer_simdgroup: bool = False
) -> tuple[str, int]:
    """Recognize a tiled matrix product and emit its cooperative TensorOps kernel.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``. The name must be a valid
        MSL identifier.
    prefer_simdgroup : bool, optional
        Emit the `simdgroup_matrix` kernel of `simdgroup.py` instead of `matmul2d`
        for the products it supports: untransposed operands, evenly tiled M and
        N, and K a multiple of 16. Other products keep `matmul2d`.

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
    Supports rank-2 matrices or matching rank-3 batches on a 2D or 3D grid, matching
    float32/float16/bfloat16 buffers, and recognized scalar epilogues. Requires one dot, one
    full-block output store, no scratch or aliases, and the full K extent in each Pallas
    tile. K is processed in internal chunks with tail guards; partial M/N edge tiles are
    masked. Rank-2 operands may be transposed; batch broadcasting is rejected.
    """
    product = _recognize(spec)
    if (
        prefer_simdgroup
        and not (product.transpose_lhs or product.transpose_rhs or product.edge_tiles)
        and simdgroup.supports(product.tm, product.tn, product.k)
    ):
        return _lower_simdgroup(product, kernel_name)
    return _lower_matmul2d(product, kernel_name)


def _recognize(spec: KernelSpec) -> _Product:
    """Match a kernel that stores one tiled product, optionally through an epilogue.
    Every equation must have a part in that; anything else raises EmitError."""
    if (
        len(spec.inputs) < 2
        or len(spec.outputs) != 1
        or spec.scratch
        or spec.aliases
        or len(spec.grid) not in (2, 3)
        or len(spec.jaxpr.constvars) != 0
        or len(spec.jaxpr.invars) != len(spec.inputs) + 1
    ):
        raise EmitError("tensorops matmul requires a full-block matmul with one output")

    dots = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "dot_general"]
    stores = [eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "swap"]
    if len(dots) != 1 or len(stores) != 1:
        raise EmitError("tensorops matmul requires one dot and one output store")
    (dot,), (store,) = dots, stores
    if len(store.invars) != 2 or store.params["tree"].flatten_up_to(()) != []:
        raise EmitError("tensorops matmul supports only a full-block output store")
    if store.invars[0] is not spec.jaxpr.invars[-1]:
        raise EmitError("tensorops matmul store must target its output ref")

    producers = _producers(spec.jaxpr)
    lhs_eqns, transpose_lhs = _matrix_read(dot.invars[0], producers, spec.jaxpr.invars[0])
    rhs_eqns, transpose_rhs = _matrix_read(dot.invars[1], producers, spec.jaxpr.invars[1])

    lhs_info, rhs_info = spec.inputs[:2]
    output_info = spec.outputs[0]
    matrices = (lhs_info, rhs_info, output_info)
    if CTYPES.get(lhs_info.dtype) not in ("float", "half", "bfloat") or any(
        info.dtype != lhs_info.dtype for info in (*spec.inputs, output_info)
    ):
        raise EmitError("TensorOps matmul requires matching float32, float16, or bfloat16 types")
    rank = len(output_info.array_shape)
    if rank not in (2, 3) or any(
        len(info.array_shape) != rank or len(info.block_shape) != rank for info in matrices
    ):
        raise EmitError("TensorOps matmul supports rank-2 matrices and rank-3 batches")
    if any(info.full_block_shape != info.block_shape for info in matrices):
        raise EmitError("TensorOps matmul requires complete matrix blocks")
    if rank == 3:
        if transpose_lhs or transpose_rhs:
            raise EmitError(
                "tensorops batched TensorOps matmul does not support transposed operands"
            )
        if lhs_info.array_shape[0] != rhs_info.array_shape[0]:
            raise EmitError("tensorops batched TensorOps operands must have matching batch sizes")

    batch = tuple(lhs_info.array_shape[:-2])  # () or (batch,)
    m, k = lhs_info.array_shape[-2:][::-1] if transpose_lhs else lhs_info.array_shape[-2:]
    kb, n = rhs_info.array_shape[-2:][::-1] if transpose_rhs else rhs_info.array_shape[-2:]
    tm, tn = output_info.block_shape[-2:]
    if (
        k != kb
        or output_info.array_shape != (*batch, m, n)
        or lhs_info.block_shape[-2:] != ((k, tm) if transpose_lhs else (tm, k))
        or rhs_info.block_shape[-2:] != ((tn, kb) if transpose_rhs else (kb, tn))
        or any(info.block_shape[:-2] != (1,) * len(batch) for info in matrices)
        or spec.grid != (*batch, -(-m // tm), -(-n // tn))
        or k <= 0
        or m <= 0
        or n <= 0
        or tm % 16
        or tn % 16
    ):
        raise EmitError("TensorOps matmul requires full-K, evenly tiled matrix operands")
    if not (
        _index_map_is(
            lhs_info, (None, 0) if transpose_lhs else ((0, None) if rank == 2 else (0, 1, None))
        )
        and _index_map_is(
            rhs_info, (1, None) if transpose_rhs else ((None, 1) if rank == 2 else (0, None, 2))
        )
        and _index_map_is(output_info, tuple(range(rank)))
    ):
        raise EmitError("TensorOps matmul requires standard row-major block maps")

    store_value = store.invars[1]
    matched = {id(dot), id(store), *map(id, lhs_eqns), *map(id, rhs_eqns)}
    if rank == 3:
        broadcast = producers.get(store_value) if isinstance(store_value, Var) else None
        if broadcast is not None and broadcast.primitive.name == "broadcast_in_dim":
            if tuple(broadcast.params["broadcast_dimensions"]) != (1, 2):
                raise EmitError(
                    "tensorops batched matmul store requires a leading singleton broadcast"
                )
            matched.add(id(broadcast))
            store_value = broadcast.invars[0]
    if not isinstance(store_value, Var):
        raise EmitError("tensorops matmul store value must be the dot result or its epilogue")
    epilogue, epilogue_eqns = _epilogue_path(store_value, dot.outvars[0], producers)
    if not epilogue and store_value is not dot.outvars[0]:
        raise EmitError("tensorops matmul store value must be the dot result or its epilogue")
    if len(matched | epilogue_eqns) != len(spec.jaxpr.eqns):
        raise EmitError("tensorops matmul jaxpr has operations outside the dot epilogue path")

    return _Product(
        spec=spec,
        dot=dot,
        gets=[eqn for eqn in spec.jaxpr.eqns if eqn.primitive.name == "get"],
        epilogue=epilogue,
        store_value=store_value,
        rank=rank,
        m=m,
        n=n,
        k=k,
        tm=tm,
        tn=tn,
        transpose_lhs=transpose_lhs,
        transpose_rhs=transpose_rhs,
    )


def _lower_matmul2d(product: _Product, kernel_name: str | None) -> tuple[str, int]:
    """Emit a recognized product with Metal 4's `matmul2d`: accumulate K chunks into
    the cooperative tensor, then store it as `_placement` decides."""
    spec, m, n, k, tm, tn = product.spec, product.m, product.n, product.k, product.tm, product.tn
    transpose_lhs, transpose_rhs = product.transpose_lhs, product.transpose_rhs
    store, cooperative_epilogue = _placement(product)
    cursor = Cursor()
    builtins: tuple[str, ...] = ()
    if store is not _Store.DEVICE:  # the per-element loop of `_store_elements`
        builtins = (
            "uint tid [[thread_index_in_threadgroup]]",
            "uint3 threads_per_group [[threads_per_threadgroup]]",
        )
        cursor.emit(
            "const uint THREADS = threads_per_group.x * threads_per_group.y * threads_per_group.z;"
        )
    operands = _bind_operands(product, cursor)
    a_offset, b_offset, c_offset = operands.offsets[0], operands.offsets[1], operands.offsets[-1]
    output = f"(arg{len(spec.inputs)} + {c_offset})"

    staged = (
        cursor.allocate("float", (tm, tn), name="dot_result", space="threadgroup")
        if store is _Store.STAGED
        else None
    )
    k_tile = min(_K_TILE, -(-k // 16) * 16)
    operation = dataclasses.replace(
        _TensorOpsMatmul.from_eqn(product.dot, _producers(spec.jaxpr), name="op", accumulate=True),
        descriptor="desc",
        k=k_tile,
    )
    operation.emit_declaration(cursor)

    m_pid, n_pid = ("x", "y") if product.rank == 2 else ("y", "z")
    valid_m = str(tm) if m % tm == 0 else f"min({tm}, {m} - (int)_pid.{m_pid} * {tm})"
    valid_n = str(tn) if n % tn == 0 else f"min({tn}, {n} - (int)_pid.{n_pid} * {tn})"

    # Operand tiles from K offset `start`, `depth` deep. Extents and strides
    # describe the stored tile, innermost first: (K, M) for rows of A, or
    # (M, K) when A is stored transposed; likewise for B.
    def lhs_view(name: str, start: str, depth: str | int) -> str:
        return _tensor(
            cursor,
            name,
            f"device {product.ctype}",
            f"(arg0 + {a_offset}{start and f' + {start} * {m if transpose_lhs else 1}'})",
            (valid_m, depth) if transpose_lhs else (depth, valid_m),
            (1, m) if transpose_lhs else (1, k),
        )

    def rhs_view(name: str, start: str, depth: str | int) -> str:
        return _tensor(
            cursor,
            name,
            f"device {product.ctype}",
            f"(arg1 + {b_offset}{start and f' + {start} * {1 if transpose_rhs else n}'})",
            (depth, valid_n) if transpose_rhs else (valid_n, depth),
            (1, k) if transpose_rhs else (1, n),
        )

    lhs_tensor, rhs_tensor = lhs_view("a", "", k), rhs_view("b", "", k)
    if staged is not None:
        destination = _tensor(cursor, "c", "threadgroup float", staged.expr, (tn, tm), (1, tn))
    else:
        destination = _tensor(
            cursor,
            "c",
            f"device {product.ctype}",
            f"arg{len(spec.inputs)}" if c_offset == "0" else output,
            (valid_n, valid_m),
            (1, n),
        )
    edge = None
    if store is _Store.MASKED:
        edge = cursor.allocate(product.ctype, (tm, tn), name="edge_result", space="threadgroup")
        _tensor(cursor, "c_edge", f"threadgroup {product.ctype}", edge.expr, (tn, tm), (1, tn))

    accumulator = operation.emit_cooperative_destination(cursor, lhs_tensor, rhs_tensor)
    with cursor.loop(f"{accumulator}.get_capacity()", "init") as index:
        cursor.emit(f"{accumulator}[{index}] = 0.0f;")
    with cursor.block(f"for (int k_start = 0; k_start < {k}; k_start += {k_tile})"):
        depth = f"min({k_tile}, {k} - k_start)"
        lhs_chunk, rhs_chunk = lhs_view("a_k", "k_start", depth), rhs_view("b_k", "k_start", depth)
        cursor.emit(f"{operation.name}.run({lhs_chunk}, {rhs_chunk}, {accumulator});")

    if cooperative_epilogue:
        with cursor.loop(f"{accumulator}.get_capacity()", "element") as index:
            # The epilogue reads only the product, so the slot index is never
            # used as a tile position.
            element = f"{accumulator}[{index}]"
            expression = _epilogue_expression(cursor, product, operands, element, index)
            cursor.emit(f"{element} = {expression};")

    def store_elements(name: str, element: Callable[[str], str]) -> None:
        """Store `element(index)` for each in-bounds index of the tile, one thread each."""
        cursor.barrier()
        with cursor.strided_loop("tid", str(tm * tn), "THREADS", name=name) as index:
            cursor.emit(f"const uint row = {index} / {tn};")
            cursor.emit(f"const uint column = {index} % {tn};")
            with cursor.block(f"if (row < {valid_m} && column < {valid_n})"):
                cursor.emit(f"{output}[row * {n} + column] = {element(index)};")

    if staged is not None:
        cursor.emit(f"{accumulator}.store({destination});")
        store_elements(
            "element",
            lambda index: _epilogue_expression(
                cursor, product, operands, product.narrowed(staged.at(index)), index
            ),
        )
    elif edge is not None:
        with cursor.block(f"if ({valid_m} == {tm} && {valid_n} == {tn})"):
            cursor.emit(f"{accumulator}.store({destination});")
        with cursor.block("else"):
            cursor.emit(f"{accumulator}.store(c_edge);")
            store_elements("edge", edge.at)
    else:
        cursor.emit(f"{accumulator}.store({destination});")
    return _kernel_source(kernel_name or spec.name, product.parameters(*builtins), cursor), (
        cursor.threadgroup_bytes
    )


def _lower_simdgroup(product: _Product, kernel_name: str | None) -> tuple[str, int]:
    """Emit a recognized product with the `simdgroup_matrix` kernel.

    The caller has checked `simdgroup.supports`, untransposed operands and even
    M and N tiling. Each lane narrows the float32 elements it holds to the output
    type, applies the epilogue and stores them.
    """
    spec, n, tn = product.spec, product.n, product.tn
    cursor = Cursor()
    operands = _bind_operands(product, cursor)
    simdgroup.emit_main_loop(
        cursor,
        a=f"(arg0 + {operands.offsets[0]})",
        b=f"(arg1 + {operands.offsets[1]})",
        k=product.k,
        n=n,
        tm=product.tm,
        tn=tn,
        ctype=product.ctype,
    )
    output = f"(arg{len(spec.inputs)} + {operands.offsets[-1]})"
    with simdgroup.accumulator_elements(cursor, product.tm, tn) as (value, row, column):
        # Parenthesized: column-bias reads append `% tn` to it.
        index = f"({row} * {tn} + {column})"
        expression = _epilogue_expression(cursor, product, operands, product.narrowed(value), index)
        cursor.emit(f"{output}[{row} * {n} + {column}] = {expression};")
    source = _kernel_source(
        kernel_name or spec.name, product.parameters(*simdgroup.PARAMETERS), cursor
    )
    return source, cursor.threadgroup_bytes


@dataclasses.dataclass(frozen=True)
class _Operands:
    """The lowered refs and reads of a product.

    Attributes
    ----------
    env : Environment
        Resolves literals.
    offsets : tuple of str
        Element offset of each operand's block, inputs then output.
    values : dict
        Value of each `get` result. Epilogue reads beyond the two dot operands are
        `(tm, tn)` tile views; a `(tn,)` read with a column block map is a column bias.
    """

    env: Environment
    offsets: tuple[str, ...]
    values: dict[Var, CVal]


def _bind_operands(product: _Product, cursor: Cursor) -> _Operands:
    """Emit block offsets and lower the full-block reads of a product."""
    spec, rank, tm, tn = product.spec, product.rank, product.tm, product.tn
    env = Environment()
    infos = (*spec.inputs, spec.outputs[0])
    offsets = tuple(block_offset(env, cursor, spec, info) for info in infos)
    refs = {
        ref: CVal(
            f"arg{index}" if offset == "0" else f"(arg{index} + {offset})",
            info.block_shape,
            msl_type(info.dtype),
            space="device",
            readonly=True,
        )
        for index, (ref, info, offset) in enumerate(zip(spec.jaxpr.invars, infos, offsets))
        if index < len(spec.inputs)
    }
    tile_shape = (tm, tn) if rank == 2 else (1, tm, tn)
    values: dict[Var, CVal] = {}
    for eqn in product.gets:
        if (
            len(eqn.invars) != 1
            or eqn.invars[0] not in refs
            or tuple(eqn.params["tree"].flatten_up_to(()))
        ):
            raise EmitError("tensorops matmul reads must be full-block gets of its inputs")
        value = refs[eqn.invars[0]]
        if _shape(eqn.outvars[0]) != value.shape:
            raise EmitError("tensorops matmul get shape does not match its input block")
        index = spec.jaxpr.invars.index(eqn.invars[0])
        info = spec.inputs[index]
        column_bias = rank == 2 and value.shape == (tn,) and _index_map_is(info, (1,))
        if index >= 2 and not column_bias:
            if value.shape != tile_shape or not _index_map_is(info, tuple(range(rank))):
                raise EmitError("tensorops matmul epilogue reads must match the output tile layout")
            value = dataclasses.replace(
                value,
                shape=(tm, tn),
                index_map=f"(($i / {tn}) * {info.array_shape[-1]} + ($i % {tn}))",
            )
        values[eqn.outvars[0]] = value
    return _Operands(env, offsets, values)


def _epilogue_expression(
    cursor: Cursor, product: _Product, operands: _Operands, element: str, index: str
) -> str:
    """Emit the epilogue for one output element and return the value to store.

    Parameters
    ----------
    element : str
        The element of the dot result.
    index : str
        The element's flat position in the `tm x tn` tile, for reads of other
        tiles and column biases.

    Raises
    ------
    EmitError
        An epilogue operation changes the tile shape or has an operand that is
        neither scalar nor shaped like the tile.
    """
    tm, tn = product.tm, product.tn
    scalars = {product.dot.outvars[0]: element}

    def scalar(atom) -> str:
        if isinstance(atom, Literal):
            literal = operands.env.val(atom)
            if literal.shape:
                raise EmitError("tensorops matmul epilogue literals must be scalar")
            return literal.expr
        if atom in scalars:
            return scalars[atom]
        value = operands.values[atom]  # `_epilogue_path` admits only gets beyond these
        if value.shape == (tn,):
            return value.at(f"({index} % {tn})")
        if value.shape != (tm, tn):
            raise EmitError("tensorops matmul epilogue operands must match the output tile")
        return value.at(index)

    for eqn in product.epilogue:
        shape = _shape(eqn.outvars[0])
        broadcast = eqn.primitive.name == "broadcast_in_dim"
        if (
            shape != (tm, tn)
            and not (product.rank == 3 and shape == (1, tm, tn))
            and not (broadcast and shape == (1, tn))
        ):
            raise EmitError(
                "tensorops matmul epilogue operations must preserve the output tile shape"
            )
        arguments = tuple(scalar(atom) for atom in eqn.invars)
        result_type = msl_type(shaped(eqn.outvars[0].aval).dtype)
        if broadcast:  # of a column bias or the batch's singleton axis: the same element
            if tuple(eqn.params["broadcast_dimensions"]) not in ((1,), (1, 2)):
                raise EmitError(
                    "tensorops matmul only supports column and singleton-batch broadcasts"
                )
            (expression,) = arguments
        else:
            expression = format_scalar(eqn.primitive.name, result_type, arguments, cursor)
        name = cursor.fresh("tensorops_epilogue")
        cursor.emit(f"{result_type} {name} = {expression};")
        scalars[eqn.outvars[0]] = name
    return scalars[product.store_value]


def _matrix_read(atom, producers, ref: Var) -> tuple[list[JaxprEqn], bool]:
    """The equations reading a dot operand from the full block of `ref`, and whether
    they transpose it: a `get`, optionally transposed, optionally squeezed of a
    leading batch axis."""
    eqns = []

    def producer(name: str) -> JaxprEqn | None:
        nonlocal atom
        eqn = producers.get(atom) if isinstance(atom, Var) else None
        if eqn is None or eqn.primitive.name != name:
            return None
        eqns.append(eqn)
        atom = eqn.invars[0]
        return eqn

    squeeze = producer("squeeze")
    if squeeze is not None and tuple(squeeze.params["dimensions"]) != (0,):
        raise EmitError("tensorops batched matmul requires a leading singleton squeeze")
    transpose = producer("transpose")
    if transpose is not None and tuple(transpose.params["permutation"]) != (1, 0):
        raise EmitError("TensorOps matmul supports only matrix transposes")
    get = producer("get")
    if get is None or len(get.invars) != 1 or get.invars[0] is not ref:
        raise EmitError("tensorops dot operands must be full-block reads of the first two refs")
    return eqns, transpose is not None


def _epilogue_path(value, dot_value, producers) -> tuple[list[JaxprEqn], set[int]]:
    """The elementwise equations between the dot and the stored `value`, in dependency
    order, and the ids of those and of the `get`s they read."""
    ordered = []
    matched: set[int] = set()
    allowed = {*ELEMENTWISE, *HELPERS, "max", "broadcast_in_dim"}

    def visit(atom):
        if atom is dot_value or isinstance(atom, Literal):
            return
        producer = producers.get(atom) if isinstance(atom, Var) else None
        if producer is None:
            raise EmitError("tensorops matmul epilogue has an unbound operand")
        if id(producer) in matched:
            return
        name = producer.primitive.name
        if name != "get":
            if name not in allowed and not typed_expression(name, "float"):
                raise EmitError(f"tensorops matmul epilogue primitive {name!r} is unsupported")
            for operand in producer.invars:
                visit(operand)
            ordered.append(producer)
        matched.add(id(producer))

    visit(value)
    return ordered, matched
