"""Kernel assembly: one MSL source per KernelSpec, through TensorOps or the
one-thread-per-program rules."""

from __future__ import annotations

import math

from palladium.emit import rules as _rules  # noqa: F401 - registers the rules
from palladium.emit.addressing import (
    BlockLayout,
    block_offset,
    constant_offset,
    element_strides,
    flat_index,
)
from palladium.emit.core import (
    PID,
    Cursor,
    CVal,
    EmitError,
    EmitStats,
    Environment,
    emit_jaxpr,
    msl_type,
)
from palladium.emit.tensorops import compile_kernel, uses_tensorops
from palladium.trace import KernelSpec

DOT_GENERAL_POLICIES = ("auto", "default", "tensorops")


def emit_msl_stats(
    spec: KernelSpec,
    kernel_name: str | None = None,
    *,
    dot_general: str = "auto",
) -> tuple[str, EmitStats]:
    """Assemble the full MSL source for a KernelSpec, with its storage stats.

    Signature convention: operands in jaxpr order (inputs then outputs)
    bound to `[[buffer(k)]]`, then `uint3 _pid [[thread_position_in_grid]]`,
    one thread per program instance. TensorOps kernels use their own
    signature.

    Parameters
    ----------
    spec : KernelSpec
        Traced kernel, from `palladium.trace`.
    kernel_name : str, optional
        Overrides `spec.name` as the MSL function name.
    dot_general : str
        "auto" tries the cooperative TensorOps lowering and falls back
        to this emitter; "tensorops" requires it; "default" skips it.

    Returns
    -------
    tuple of (str, EmitStats)
        Complete, self-contained MSL source, and the per-instance
        storage the kernel declares.

    Raises
    ------
    EmitError
        For grids over rank 3 or unsupported addressing forms.
    UnsupportedPrimitiveError
        If the kernel stages a primitive with no registered rule.
    """
    if dot_general not in DOT_GENERAL_POLICIES:
        raise ValueError("dot_general must be 'auto', 'default', or 'tensorops'")
    if dot_general != "default" and uses_tensorops(spec, dot_general):
        try:
            source, threadgroup_bytes = compile_kernel(spec, kernel_name)
        except EmitError:
            # "auto" only takes dots the cooperative lowering recognizes;
            # anything else keeps the one-thread-per-program emitter.
            if dot_general != "auto":
                raise
        else:
            return source, EmitStats(thread_bytes=0, threadgroup_bytes=threadgroup_bytes)

    source, stats = _assemble(spec, kernel_name, fuse_loads=False)
    if stats.thread_bytes >= _REGISTER_BYTES:
        source, stats = _assemble(spec, kernel_name, fuse_loads=True)
    return source, stats


# Per-thread storage above which copied input blocks spill out of registers.
# Below it, copying a block first measured faster than reading it in place;
# above it, reading in place won by up to 2.9x (M2, blocked elementwise).
_REGISTER_BYTES = 512


def _assemble(
    spec: KernelSpec, kernel_name: str | None, *, fuse_loads: bool
) -> tuple[str, EmitStats]:
    name = kernel_name or spec.name
    if len(spec.grid) > 3:
        raise EmitError(f"grid {spec.grid} has rank > 3; Metal grids are 3D")

    operands = list(spec.inputs) + list(spec.outputs)
    n_in = len(spec.inputs)
    params = []
    for k, info in enumerate(operands):
        qual = "device" if k >= n_in else "const device"
        ctype = msl_type(info.dtype)
        params.append(f"{qual} {ctype}* arg{k} [[buffer({k})]]")
    params.append("uint3 _pid [[thread_position_in_grid]]")

    env = Environment(
        no_stream_refs=frozenset(
            spec.jaxpr.invars[k] for i, j in spec.aliases for k in (i, n_in + j)
        ),
        fuse_loads=fuse_loads,
    )
    cursor = Cursor()
    ref_vals: list[CVal] = []
    # Aliased inputs share their buffer with an output; dropping readonly
    # forces gets to copy instead of binding a view that would observe
    # the in-place write.
    aliased_inputs = {i for i, _ in spec.aliases}
    for k, info in enumerate(operands):
        qual = "device" if k >= n_in else "const device"
        ctype = msl_type(info.dtype)
        layout = BlockLayout.from_info(info)
        full = layout.shape
        strided = any(b != a for b, a in zip(full[1:], info.array_shape[1:]))
        edge = any(a % b for a, b in zip(info.array_shape, full))
        if strided or edge:
            dims = info.full_block_shape
            pids = [
                CVal(f"(int){PID[d]}", (), "int")
                for d in range(len(info.index_map_jaxpr.jaxpr.invars))
            ]
            origins = emit_jaxpr(env, cursor, info.index_map_jaxpr.jaxpr, pids)
            logical_strides = iter(element_strides(info.block_shape))
            coords = []
            for dim, size, origin in zip(dims, full, origins, strict=True):
                local = "0" if dim is None else f"(($i / {next(logical_strides)}) % {size})"
                coords.append(f"(int({origin.expr}) * {size} + int({local}))")
            address = flat_index(list(zip(coords, element_strides(info.array_shape))))
            valid = " && ".join(
                f"({c} >= 0 && {c} < {a})" for c, a in zip(coords, info.array_shape)
            )
            ref_vals.append(
                CVal(
                    f"arg{k}",
                    info.block_shape or (1,),
                    ctype,
                    space="device",
                    readonly=k < n_in and k not in aliased_inputs,
                    index_map=address,
                    valid=valid,
                )
            )
            continue
        constant = constant_offset(info)
        offset = constant if constant is not None else block_offset(env, cursor, spec, info)
        if offset == "0":
            ptr = f"arg{k}"
        else:
            ptr = f"arg{k}_offset"
            cursor.emit(f"{qual} {ctype}* {ptr} = arg{k} + {offset};")

        # Scalar refs (shape ()) are addressed as one-element arrays.
        ref_vals.append(
            CVal(
                expr=ptr,
                shape=info.block_shape or (1,),
                ctype=ctype,
                space="device",
                readonly=k < n_in and k not in aliased_inputs,
                align=layout.alignment() if constant is None else int(constant),
            )
        )

    for k, info in enumerate(spec.scratch):
        ctype = msl_type(info.dtype)
        shape = info.shape
        size = math.prod(info.shape)
        cursor.account(ctype, size, "thread")
        scratch_op = f"{ctype} scratch{k}"
        scratch_op += f"[{size}];" if shape else ";"
        cursor.emit(scratch_op)
        ref_vals.append(
            CVal(
                expr=f"scratch{k}",
                shape=shape,
                ctype=ctype,
                readonly=False,
                align=0,
            )
        )

    n_refs = len(operands) + len(spec.scratch)
    if len(spec.jaxpr.invars) != n_refs:
        raise EmitError(
            f"kernel has {len(spec.jaxpr.invars)} refs but spec carries {n_refs} operands"
        )
    emit_jaxpr(env, cursor, spec.jaxpr, ref_vals)

    head = f"kernel void {name}(\n    " + ",\n    ".join(params) + ")\n{"
    source = "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            *cursor.helpers.values(),
            head,
            *cursor.lines,
            "}",
            "",
        ]
    )
    return source, EmitStats(
        thread_bytes=cursor.thread_bytes,
        threadgroup_bytes=cursor.threadgroup_bytes,
    )


def emit_msl(
    spec: KernelSpec,
    kernel_name: str | None = None,
    *,
    dot_general: str = "auto",
) -> str:
    """Assemble the full MSL source for a KernelSpec; see `emit_msl_stats`."""
    return emit_msl_stats(spec, kernel_name, dot_general=dot_general)[0]
