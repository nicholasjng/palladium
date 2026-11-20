"""Kernel assembly: one MSL source per KernelSpec, through TensorOps or the
one-thread-per-program rules."""

from __future__ import annotations

import math

from palladium.device import prefers_simdgroup_matrix
from palladium.emit import rules as _rules  # noqa: F401 - registers the rules
from palladium.emit.addressing import (
    block_strides,
    element_strides,
    flat_index,
    full_block_shape,
)
from palladium.emit.core import (
    PID,
    REGISTER_BYTES,
    Cursor,
    CVal,
    EmitError,
    EmitStats,
    Environment,
    emit_jaxpr,
    msl_type,
)
from palladium.emit.mesh import lower_mesh_program
from palladium.emit.tensorops import compile_kernel, uses_tensorops
from palladium.emit.threadgroup import lower_row_program
from palladium.trace import KernelSpec

DOT_GENERAL_POLICIES = ("auto", "default", "tensorops", "simdgroup")
PROGRAM_SCOPES = ("thread", "threadgroup")


def emit_msl_stats(
    spec: KernelSpec,
    kernel_name: str | None = None,
    *,
    dot_general: str = "auto",
    program: str = "thread",
) -> tuple[str, EmitStats]:
    """`emit_msl` with the storage the kernel declares; arguments and errors are the same.

    The one-thread-per-program emitter re-emits with eligible input loads
    fused into their consumers once its thread-local storage reaches
    `REGISTER_BYTES`. Cooperative lowerings report threadgroup storage and no
    thread-local bytes. The counts cover declarations only: not compiler
    temporaries, liveness, or register use inside MPP operations.
    """
    if dot_general not in DOT_GENERAL_POLICIES:
        raise ValueError("dot_general must be 'auto', 'default', 'tensorops', or 'simdgroup'")
    if program not in PROGRAM_SCOPES:
        raise ValueError("program must be 'thread' or 'threadgroup'")
    if spec.simdgroups is not None:
        source, threadgroup_bytes = lower_mesh_program(spec, kernel_name)
        return source, EmitStats(thread_bytes=0, threadgroup_bytes=threadgroup_bytes)
    if program == "threadgroup":
        source, threadgroup_bytes = lower_row_program(spec, kernel_name)
        return source, EmitStats(thread_bytes=0, threadgroup_bytes=threadgroup_bytes)
    if dot_general != "default" and uses_tensorops(spec, dot_general):
        prefer_simdgroup = dot_general == "simdgroup" or (
            dot_general == "auto" and prefers_simdgroup_matrix()
        )
        try:
            source, threadgroup_bytes = compile_kernel(
                spec, kernel_name, prefer_simdgroup=prefer_simdgroup
            )
        except EmitError:
            # "auto" only takes dots the cooperative lowering recognizes;
            # anything else keeps the one-thread-per-program emitter.
            if dot_general != "auto":
                raise
        else:
            return source, EmitStats(thread_bytes=0, threadgroup_bytes=threadgroup_bytes)

    source, stats = _assemble(spec, kernel_name, fuse_loads=False)
    if stats.thread_bytes >= REGISTER_BYTES:
        source, stats = _assemble(spec, kernel_name, fuse_loads=True)
    return source, stats


def _assemble(
    spec: KernelSpec, kernel_name: str | None, *, fuse_loads: bool
) -> tuple[str, EmitStats]:
    """One-thread-per-program MSL and its declared storage; `fuse_loads` reads eligible input blocks in place."""
    name = kernel_name or spec.name
    if len(spec.grid) > 3:
        raise EmitError(f"grid {spec.grid} has rank > 3; Metal grids are 3D")

    operands = list(spec.inputs) + list(spec.outputs)
    n_in = len(spec.inputs)
    params = []

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
        params.append(f"{qual} {ctype}* arg{k} [[buffer({k})]]")
        full = full_block_shape(info)
        steps = block_strides(info)
        strided = any(b != a for b, a in zip(full[1:], info.array_shape[1:]))
        edge = any(a % b for a, b in zip(info.array_shape, full))
        pids = [
            CVal(f"(int){PID[d]}", (), "int") for d in range(len(info.index_map_jaxpr.jaxpr.invars))
        ]
        origins = emit_jaxpr(env, cursor, info.index_map_jaxpr.jaxpr, pids)
        # The provable multiple of every block origin: the offset itself when
        # the index map is constant, the common divisor of the steps otherwise.
        if all(origin.expr.lstrip("-").isdigit() for origin in origins):
            align = sum(int(origin.expr) * step for origin, step in zip(origins, steps))
            offset = str(align)
        else:
            align = math.gcd(*steps)
            offset = flat_index(zip((origin.expr for origin in origins), steps, strict=True))
        if offset == "0":
            ptr = f"arg{k}"
        else:
            ptr = f"arg{k}_offset"
            cursor.emit(f"{qual} {ctype}* {ptr} = arg{k} + {offset};")
        index_map = valid = None
        if strided or edge:
            # The block's rows are not adjacent in the array, or it reaches
            # past the array: address elements by their block coordinates.
            logical_strides = iter(element_strides(info.block_shape))
            local = [
                "0" if dim is None else f"(($i / {next(logical_strides)}) % {size})"
                for dim, size in zip(info.full_block_shape, full, strict=True)
            ]
            index_map = flat_index(zip(local, element_strides(info.array_shape)))
        if edge:
            # Only a block past the array's end has elements to skip.
            valid = " && ".join(
                f"({c} >= 0 && {c} < {a})"
                for c, a in zip(
                    (
                        f"(int({origin.expr}) * {size} + int({index}))"
                        for origin, size, index in zip(origins, full, local, strict=True)
                    ),
                    info.array_shape,
                )
            )

        # Scalar refs (shape ()) are addressed as one-element arrays.
        ref_vals.append(
            CVal(
                expr=ptr,
                shape=info.block_shape or (1,),
                ctype=ctype,
                space="device",
                readonly=k < n_in and k not in aliased_inputs,
                align=align,
                index_map=index_map,
                valid=valid,
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

    params.append("uint3 _pid [[thread_position_in_grid]]")
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
    program: str = "thread",
) -> str:
    """Generate a complete Metal kernel from a traced Pallas specification.

    Parameters
    ----------
    spec : KernelSpec
        Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
    kernel_name : str or None, optional
        MSL function name override; None uses ``spec.name``. The name must be a valid
        MSL identifier.
    dot_general : {"auto", "default", "tensorops", "simdgroup"}, optional
        Use cooperative TensorOps when recognized, force primitive emission, require
        TensorOps, or require them with `simdgroup_matrix` matmuls,
        respectively. The default is "auto", which also prefers `simdgroup_matrix`
        on M1 to M4 GPUs (Apple families 7 to 9), where it is faster than
        `matmul2d`.
    program : {"thread", "threadgroup"}, optional
        "thread" (default) runs each Pallas program as one Metal thread, or
        one threadgroup when a dot lowers cooperatively. "threadgroup" runs
        each program as one threadgroup of rows (`emit/threadgroup.py`);
        kernels it cannot lower raise EmitError.

    Returns
    -------
    str
        Complete MSL source; emission does not compile or dispatch the kernel.

    Raises
    ------
    EmitError
        The kernel uses unsupported shapes, addressing, effects, or primitive
        parameters.
    UnsupportedPrimitiveError
        A staged primitive has no registered lowering rule.
    ValueError
        The dot policy is unknown.

    Notes
    -----
    Primitive emission maps one Metal thread to each program. Recognized TensorOps kernels
    map one threadgroup to each program; launch them with the matching geometry. "auto"
    falls back when cooperative recognition raises EmitError, while "tensorops" propagates
    that error for kernels containing dots. No device compilation occurs.

    See Also
    --------
    emit_msl_stats : Return source together with declared storage estimates.
    """
    return emit_msl_stats(spec, kernel_name, dot_general=dot_general, program=program)[0]
