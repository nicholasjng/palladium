"""Extract the kernel jaxpr from Pallas: trace the wrapped `pl.pallas_call`
or `pl.kernel` with `jax.make_jaxpr` and repackage its equation into a
KernelSpec. `grid_mapping`/`block_mapping` fields are version-sensitive.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.experimental.pallas as pl
import numpy as np
from jax._src.pallas import core as pallas_core  # get_grid_mapping has no public name
from jax.extend.core import (
    ClosedJaxpr,
    Jaxpr,
    JaxprEqn,
    Literal,
    Var,
    jaxprs_in_params,
    subjaxprs,
)

from palladium import effects
from palladium.errors import TraceError

__all__ = [
    "BlockInfo",
    "KernelSpec",
    "ScratchInfo",
    "spec_from_params",
    "trace",
]


@dataclasses.dataclass(frozen=True)
class ScratchInfo:
    """Describe an uninitialized scratch buffer private to one Pallas program.

    Attributes
    ----------
    shape : tuple of int
        Logical buffer dimensions; () denotes scalar scratch.
    dtype : numpy.dtype
        Element dtype, which must be supported by the selected emitter.
    space : str, optional
        "thread" (default) for storage private to each program, or "threadgroup" for
        `plmtl.THREADGROUP` tiles shared by a mesh program's threads.

    Notes
    -----
    Scratch has no caller-owned array and must be written before it is read. The primitive
    emitter allocates it in thread-local storage.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    space: str = "thread"


@dataclasses.dataclass(frozen=True)
class BlockInfo:
    """Describe the logical tile and physical array layout of one kernel operand.

    Attributes
    ----------
    block_shape : tuple of int
        Kernel-visible shape with squeezed axes removed; () denotes a scalar block.
    array_shape : tuple of int
        Full backing array shape in row-major order.
    dtype : numpy.dtype
        Shared element dtype of the block and backing array.
    index_map_jaxpr : ClosedJaxpr
        Staged map from grid coordinates to block indices, measured in blocks rather
        than elements.
    full_block_shape : tuple of int or None
        Block dimensions in original array-axis order. None marks a squeezed axis, which
        occupies one physical element.

    Notes
    -----
    Partial edge tiles and strided blocks retain the backing array shape so emission can
    guard loads and stores. Only blocked and squeezed Pallas dimensions are normalized by
    tracing.
    """

    block_shape: tuple[int, ...]
    array_shape: tuple[int, ...]
    dtype: np.dtype
    index_map_jaxpr: ClosedJaxpr
    # Unsqueezed layout, preserving the original axis positions.
    full_block_shape: tuple[int | None, ...]


@dataclasses.dataclass(frozen=True)
class KernelSpec:
    """Hold the backend-independent kernel body and layout metadata consumed by emission.

    Attributes
    ----------
    name : str
        Kernel name used as the default MSL function identifier.
    jaxpr : Jaxpr
        Stateful kernel body with Ref operands in input, output, then scratch order.
    grid : tuple of int
        Static program grid; tracing normalizes a gridless call to (1,). Metal emission
        supports at most three axes.
    inputs : tuple of BlockInfo
        Input operands in jaxpr order.
    outputs : tuple of BlockInfo
        Output operands in jaxpr order.
    scratch : tuple of ScratchInfo
        Per-program scratch operands, after inputs and outputs.
    aliases : tuple of tuple of int, optional
        Validated (input index, output index) pairs sharing a buffer. Defaults to no
        aliases.
    simdgroups : int or None, optional
        SIMD groups per threadgroup for a `pl.kernel` over a `plmtl.Mesh`, whose programs
        are threadgroups over whole arrays; None for a `pallas_call`.
    sequential : tuple of int, optional
        Grid axes whose programs run in order inside one thread, with scratch kept
        between them, instead of as parallel threads. Empty by default.
    prefetch : int, optional
        How many leading inputs are the scalar operands of a
        ``PrefetchScalarGridSpec``: whole arrays that the kernel reads and that the
        index maps of the other operands take after the grid indices. Default 0.

    Notes
    -----
    Prefer ``trace`` to constructing this record directly: the dataclass itself does not
    validate layouts, aliases, or write safety. Alias validation requires identical layouts
    and input reads before output writes.
    """

    name: str
    jaxpr: Jaxpr
    grid: tuple[int, ...]
    inputs: tuple[BlockInfo, ...]
    outputs: tuple[BlockInfo, ...]
    scratch: tuple[ScratchInfo, ...]
    aliases: tuple[tuple[int, int], ...] = ()
    simdgroups: int | None = None
    sequential: tuple[int, ...] = ()
    prefetch: int = 0


def _block_dim(dim: Any) -> int | None:
    """Normalize a block extent; None for a squeezed dimension, which vanishes
    from the visible block. pl.Element, pl.Indirect and pl.BoundedSlice also
    carry `block_size` but index differently, so they are rejected rather than
    lowered with wrong offsets."""
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    if isinstance(dim, pl.Squeezed):
        return None
    if isinstance(dim, pl.Blocked):
        return int(dim.block_size)
    raise TraceError(
        f"unsupported block dim type {type(dim).__name__}: only int, "
        "pl.Blocked, and pl.Squeezed dims are lowered (pl.Element, "
        "pl.Indirect, and pl.BoundedSlice indexing is unimplemented)"
    )


def _block_infos(block_mappings: list[Any]) -> tuple[BlockInfo, ...]:
    """Preserve each block mapping's visible and unsqueezed shapes, dtype, and staged index map."""
    infos = []
    for bm in block_mappings:
        dims = [_block_dim(d) for d in bm.block_shape]
        infos.append(
            BlockInfo(
                block_shape=tuple(d for d in dims if d is not None),
                array_shape=tuple(bm.array_aval.shape),
                dtype=np.dtype(bm.array_aval.dtype),
                index_map_jaxpr=bm.index_map_jaxpr,
                full_block_shape=tuple(dims),
            )
        )
    return tuple(infos)


def _whole_array(aval: Any, grid_rank: int) -> BlockInfo:
    """An operand every program sees whole: one block the size of the array."""
    shape = tuple(int(d) for d in aval.shape)
    origin = jax.make_jaxpr(lambda *grid: (0,) * len(shape))(*(0,) * grid_rank)
    return BlockInfo(shape, shape, np.dtype(aval.dtype), origin, shape)


def _validate_aliases(
    jaxpr: Jaxpr,
    aliases: tuple[tuple[int, int], ...],
    inputs: tuple[BlockInfo, ...],
    outputs: tuple[BlockInfo, ...],
) -> None:
    """Sharing one buffer is transparent only when both refs slice it
    identically and every read of the input precedes any write of the
    output (Pallas keeps the input's pre-call values visible throughout).
    Order is checked on equation effects, which cover sub-jaxprs.
    """
    for i, j in aliases:
        a, b = inputs[i], outputs[j]
        if (
            a.array_shape != b.array_shape
            or a.dtype != b.dtype
            or a.block_shape != b.block_shape
            or a.full_block_shape != b.full_block_shape
            or str(a.index_map_jaxpr) != str(b.index_map_jaxpr)
        ):
            raise TraceError(
                f"input_output_aliases pair ({i}, {j}): input and output "
                "must have identical array shape, dtype, block shape, and "
                "BlockSpec index map to share one buffer"
            )
        x_var = jaxpr.invars[i]
        o_var = jaxpr.invars[len(inputs) + j]
        last_read = max(
            (n for n, e in enumerate(jaxpr.eqns) if effects.eqn_reads_ref(e, x_var)),
            default=-1,
        )
        first_write = min(
            (n for n, e in enumerate(jaxpr.eqns) if effects.eqn_writes_ref(e, o_var)),
            default=len(jaxpr.eqns),
        )
        if last_read >= first_write:
            raise TraceError(
                f"input_output_aliases pair ({i}, {j}): input {i} is read "
                f"after (or while) output {j} is written. The refs share "
                "one buffer on the GPU, so that read would observe the "
                "in-place write; reorder the kernel to read all its input "
                "before the first write to the aliased output"
            )


def _depends_on(jaxpr: Jaxpr, seeds: set[Var], targets: Sequence[Var | Literal]) -> bool:
    """Whether any of `targets` depends on a var in `seeds` over top-level
    eqns. Coarse across sub-jaxprs: any tainted invar taints all outvars."""
    tainted = set(seeds)
    for eqn in jaxpr.eqns:
        if any(isinstance(v, Var) and v in tainted for v in eqn.invars):
            tainted.update(eqn.outvars)
    return any(isinstance(v, Var) and v in tainted for v in targets)


def _map_uses_axis(index_map: ClosedJaxpr, axis: int) -> bool:
    """Check coarse dependency of map outputs on a grid axis; an absent axis is unused."""
    inner = index_map.jaxpr
    if axis >= len(inner.invars):
        return False
    seed = inner.invars[axis]
    return seed in inner.outvars or _depends_on(inner, {seed}, list(inner.outvars))


def _inner_ref(eqn: JaxprEqn, ref: Var) -> list[tuple[Jaxpr, Var | None]]:
    """The bodies of a control-flow equation, each with the variable `ref` is inside it
    (None if the body does not take it)."""
    name, params = eqn.primitive.name, eqn.params
    if name == "cond":
        bodies = [(branch.jaxpr, eqn.invars[1:]) for branch in params["branches"]]
    elif name == "while":
        split = params["cond_nconsts"]
        carries = eqn.invars[split + params["body_nconsts"] :]
        bodies = [
            (params["cond_jaxpr"].jaxpr, [*eqn.invars[:split], *carries]),
            (params["body_jaxpr"].jaxpr, eqn.invars[split:]),
        ]
    elif name == "scan":
        bodies = [(params["jaxpr"].jaxpr, eqn.invars)]
    else:
        bodies = [
            (getattr(child, "jaxpr", child), eqn.invars) for child in jaxprs_in_params(params)
        ]
    found = []
    for body, operands in bodies:
        inner = [v for v, outer in zip(body.invars, operands, strict=False) if outer is ref]
        found.append((body, inner[0] if inner else None))
    return found


def _first_touch(jaxpr: Jaxpr, ref: Var) -> str | None:
    """Whether a program's first use of `ref` is a "read" or a "write"; None if it has none.

    A conditional counts as a write only if every branch writes first, and a loop
    as its body does, taking it to run at least once.
    """
    for eqn in jaxpr.eqns:
        if not (effects.eqn_reads_ref(eqn, ref) or effects.eqn_writes_ref(eqn, ref)):
            continue
        bodies = _inner_ref(eqn, ref)
        if not bodies:
            # `get` reads; `addupdate` reads what it adds to; `swap` writes.
            return "write" if eqn.primitive.name == "swap" else "read"
        touches = [_first_touch(body, inner) for body, inner in bodies if inner is not None]
        if "read" in touches:
            return "read"
        if eqn.primitive.name != "cond" and "write" in touches:
            return "write"
        if touches and len(touches) == len(bodies) and all(t == "write" for t in touches):
            return "write"
    return None


def _sequential_axes(
    jaxpr: Jaxpr, n_operands: int, grid: tuple[int, ...], compiler_params: Any
) -> tuple[int, ...]:
    """The grid axes that must run in order, because the kernel reads scratch it has
    not written: values an earlier program along the grid left there.

    Mosaic runs a grid in order with scratch kept between programs, and its kernels
    accumulate that way: zero the scratch at the first step of an axis, add to it at
    every step, write the output at the last. Palladium runs programs as parallel
    threads with scratch of their own. With `dimension_semantics` the axes not
    marked "parallel" run in order; another backend's parameters without them mean
    Mosaic's default, every axis in order.
    """
    if not any(_first_touch(jaxpr, ref) == "read" for ref in jaxpr.invars[n_operands:]):
        return ()
    semantics = getattr(compiler_params, "dimension_semantics", None)
    ours = getattr(compiler_params, "BACKEND", None) == "palladium"
    if semantics is None and (compiler_params is None or ours):
        raise TraceError(
            "the kernel reads scratch before writing it, which only has a meaning if "
            "programs run in order and scratch persists between them. Palladium runs "
            "programs as parallel threads, each with scratch of its own. Write the "
            "scratch first, or pass compiler_params=palladium.CompilerParams("
            'dimension_semantics=...) with "arbitrary" for the grid axes to run in '
            'order and "parallel" for the others'
        )
    if semantics is None:
        return tuple(range(len(grid)))
    if len(semantics) != len(grid):
        raise TraceError(f"dimension_semantics {semantics} does not match the grid {grid}")
    return tuple(axis for axis, kind in enumerate(semantics) if kind != "parallel")


def _validate_parallel_writes(
    jaxpr: Jaxpr,
    outputs: tuple[BlockInfo, ...],
    grid: tuple[int, ...],
    n_in: int,
    sequential: tuple[int, ...] = (),
) -> None:
    """Reject the provable write race: a grid axis that neither the output's
    index map nor any write index depends on. A write inside control flow
    counts as indexed by the axis when the control-flow equation takes an
    operand derived from `program_id(axis)` or calls it in its body.
    Non-injective maps that use the axis pass unchecked.
    """
    for j, info in enumerate(outputs):
        o_var = jaxpr.invars[n_in + j]
        writes = [e for e in jaxpr.eqns if effects.eqn_writes_ref(e, o_var)]
        for axis, extent in enumerate(grid):
            if not writes or extent <= 1 or _map_uses_axis(info.index_map_jaxpr, axis):
                continue
            if axis in sequential:  # one thread walks this axis: no race
                continue
            pid_vars = {
                out
                for e in jaxpr.eqns
                if e.primitive.name == "program_id" and e.params["axis"] == axis
                for out in e.outvars
            }
            for eqn in writes:
                if eqn.primitive.name == "swap":
                    indexed = _depends_on(jaxpr, pid_vars, list(eqn.invars[2:]))
                else:
                    indexed = _depends_on(jaxpr, pid_vars, list(eqn.invars)) or any(
                        _calls_program_id(child, axis) for child in jaxprs_in_params(eqn.params)
                    )
                if not indexed:
                    raise TraceError(
                        f"output {j} is written identically by all "
                        f"{extent} program instances along grid axis "
                        f"{axis}: neither its BlockSpec index map nor the "
                        "write index depends on that axis, and instances "
                        "run as parallel threads, so the writes race. Make "
                        f"the output BlockSpec index map use grid axis "
                        f"{axis}, or index the write with pl.program_id({axis})"
                    )


def _calls_program_id(jaxpr: Jaxpr, axis: int) -> bool:
    """Find a program_id call for an axis in this jaxpr or any nested region."""
    return any(
        eqn.primitive.name == "program_id" and eqn.params["axis"] == axis for eqn in jaxpr.eqns
    ) or any(_calls_program_id(child, axis) for child in subjaxprs(jaxpr))


def trace(pallas_fn: Callable, *example_args) -> KernelSpec:
    """Extract one Pallas kernel specification using only input shapes and dtypes.

    Parameters
    ----------
    pallas_fn : callable
        Callable returned by ``pl.pallas_call``, or a wrapper containing exactly one
        such call. JIT wrappers are traversed.
    *example_args : array-like or jax.ShapeDtypeStruct
        Inputs fixing shapes and dtypes; no input data is read or dispatched.

    Returns
    -------
    KernelSpec
        Kernel jaxpr, normalized grid, operand layouts, scratch, and validated aliases.

    Raises
    ------
    TraceError
        No call or multiple calls are found, or the kernel uses scalar prefetch,
        unsupported block dimensions, foreign effects, invalid aliases, provably
        colliding writes, or a name that is not an identifier.

    Notes
    -----
    Only JIT wrappers are searched for calls; arbitrary nested control-flow wrappers are not
    traversed. Write checks reject obvious collisions but do not prove index-map
    injectivity. This adapter reads private Pallas metadata, which changes between JAX
    releases.
    """
    closed = jax.make_jaxpr(pallas_fn)(*example_args)
    eqns = _pallas_call_eqns(closed.jaxpr)
    if not eqns:
        raise TraceError(
            "no pallas_call equation found; pass the callable returned by "
            "pl.pallas_call, or a function that invokes one"
        )
    if len(eqns) > 1:
        raise TraceError(
            f"found {len(eqns)} pallas_call equations; palladium handles one "
            "kernel at a time, trace them separately"
        )
    (eqn,) = eqns
    if eqn.primitive.name == "mpmd_map":
        return _mesh_spec(eqn)
    return spec_from_params(eqn.params)


def _mesh_spec(eqn: JaxprEqn) -> KernelSpec:
    """A `pl.kernel` on a `plmtl.Mesh`, mapped onto a grid of whole-array programs the
    way JAX's own fallback lowering maps a mesh onto `pallas_call`."""
    from palladium.plmtl import Mesh

    params = eqn.params
    if len(params["meshes"]) != 1 or not isinstance(params["meshes"][0], Mesh):
        raise TraceError("palladium runs pl.kernel on one palladium.plmtl.Mesh")
    (mesh,) = params["meshes"]
    (jaxpr,) = params["jaxprs"]
    if mesh.simdgroups != 4:
        raise TraceError("plmtl meshes require 4 SIMD groups per threadgroup")
    if not 1 <= len(mesh.grid) <= 3:
        raise TraceError("plmtl meshes have one to three grid axes")
    out_avals = params["out_avals"]
    n_in, n_out = len(eqn.invars), len(out_avals)
    refs = [v.aval.inner_aval for v in jaxpr.invars[: n_in + n_out]]
    scratch = [v.aval for v in jaxpr.invars[n_in + n_out :]]
    any_space = pl.MemorySpace.ANY
    grid_spec = pl.GridSpec(
        grid=tuple(mesh.grid),
        in_specs=tuple(pl.BlockSpec(memory_space=any_space) for _ in range(n_in)),
        out_specs=tuple(pl.BlockSpec(memory_space=any_space) for _ in range(n_out)),
        scratch_shapes=tuple(pl.MemoryRef(a.inner_aval, a.memory_space) for a in scratch),
    )
    _, grid_mapping = pallas_core.get_grid_mapping(
        grid_spec,
        refs[:n_in],
        jax.tree_util.tree_structure(tuple(range(n_in))),
        [f"arg{i}" for i in range(n_in)],
        refs[n_in:],
        jax.tree_util.tree_structure(tuple(range(n_out))),
        [f"out{i}" for i in range(n_out)],
        debug=False,
    )
    spec = spec_from_params(
        {
            "jaxpr": jaxpr,
            "grid_mapping": grid_mapping,
            "name": params.get("name"),
            "input_output_aliases": tuple(params["input_output_aliases"].items()),
        }
    )
    return dataclasses.replace(spec, simdgroups=mesh.simdgroups)


def spec_from_params(pallas_params: Mapping[str, Any]) -> KernelSpec:
    """Normalize the parameters of a staged Pallas equation into a kernel specification.

    Parameters
    ----------
    pallas_params : mapping of str to object
        Parameters of a staged ``pallas_call``, including ``jaxpr`` and ``grid_mapping``.

    Returns
    -------
    KernelSpec
        Validated specification with a gridless call normalized to one program.

    Raises
    ------
    TraceError
        Unsupported effects, prefetch, block dimensions, aliases, obvious parallel
        write collisions, or a kernel name that is not an identifier are detected.
    KeyError
        Required JAX equation parameters are absent.

    Notes
    -----
    This is the common adapter for offline tracing and the `metal` lowering. It consumes JAX
    internal metadata rather than user-facing Pallas keyword arguments; it does not perform
    MSL primitive validation.
    """
    params = pallas_params
    grid_mapping = params["grid_mapping"]

    kernel_jaxpr = params["jaxpr"]

    foreign = effects.foreign_effects(kernel_jaxpr)
    if foreign:
        names = ", ".join(sorted({type(e).__name__ for e in foreign}))
        raise TraceError(
            f"kernel body has effects palladium cannot perform on the GPU: "
            f"{names}. Only Ref reads and writes are supported; remove "
            "debug prints, callbacks, and other host-side effects from the "
            "kernel."
        )

    prefetch = grid_mapping.num_index_operands
    n_in, n_out = grid_mapping.num_inputs, grid_mapping.num_outputs
    scratch_bufs = grid_mapping.scratch_avals
    mappings = list(grid_mapping.block_mappings)

    grid = tuple(int(g) for g in grid_mapping.grid)
    if not grid:
        grid = (1,)

    # Scalar prefetch operands lead the kernel's refs: whole arrays, at one place.
    scalars = tuple(
        _whole_array(v.aval.inner_aval, len(grid)) for v in kernel_jaxpr.invars[:prefetch]
    )
    inputs = scalars + _block_infos(mappings[:n_in])
    outputs = _block_infos(mappings[n_in : n_in + n_out])
    n_in += prefetch
    aliases = tuple((int(i), int(j)) for i, j in params.get("input_output_aliases") or ())
    if aliases:
        _validate_aliases(kernel_jaxpr, aliases, inputs, outputs)
    sequential = _sequential_axes(kernel_jaxpr, n_in + n_out, grid, params.get("compiler_params"))
    _validate_parallel_writes(kernel_jaxpr, outputs, grid, n_in, sequential)

    name = params.get("name") or "palladium_kernel"
    if not re.fullmatch(r"[A-Za-z_]\w*", name, re.ASCII):
        raise TraceError(
            f"kernel name {name!r} is not a valid identifier: it names the Metal function, "
            "so use letters, digits and underscores, not starting with a digit"
        )

    return KernelSpec(
        name=name,
        jaxpr=kernel_jaxpr,
        grid=grid,
        inputs=inputs,
        outputs=outputs,
        scratch=tuple(
            ScratchInfo(
                shape=tuple(int(d) for d in aval.shape),
                dtype=np.dtype(aval.dtype),
                space="threadgroup"
                if str(getattr(aval, "memory_space", None)) == "threadgroup"
                else "thread",
            )
            for aval in scratch_bufs
        ),
        aliases=aliases,
        sequential=sequential,
        prefetch=prefetch,
    )


def _pallas_call_eqns(jaxpr: Jaxpr) -> list[JaxprEqn]:
    """pallas_call and mpmd_map (`pl.kernel`) equations in `jaxpr`, looking through
    jit wrappers."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in ("pallas_call", "mpmd_map"):
            found.append(eqn)
        elif eqn.primitive.name == "jit":
            found.extend(_pallas_call_eqns(eqn.params["jaxpr"].jaxpr))
    return found
