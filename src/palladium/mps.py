"""Palladium as the Pallas backend for the jax-mps ``mps`` platform.

Importing palladium wraps the ``pallas_call`` lowering: on ``mps`` it emits
a ``palladium.dispatch`` StableHLO custom call whose ``backend_config`` is
``MpsDispatchDescriptor.to_json()``, and jax-mps's handler builds an MLX
kernel from it on its own Metal stream. Every other platform keeps JAX's
own lowering. Metal-side options travel as ``compiler_params``::

    pl.pallas_call(kernel, out_shape=..., compiler_params=palladium.CompilerParams(
        dot_general="tensorops", threadgroup=128))
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from jax._src import effects as jax_effects
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.pallas.pallas_call import pallas_call_p
from metal_runtime import MathMode

from palladium.emit import emit_msl
from palladium.launch import CompilerParams, check_threadgroup, launch_geometry
from palladium.trace import KernelSpec, spec_from_params

__all__ = ["MpsDispatchDescriptor"]


# An ordinary StableHLO custom-call target, not a jax.ffi target: jax-mps
# owns the buffers and encodes the dispatch on its own Metal stream.
_CUSTOM_CALL_TARGET = "palladium.dispatch"
_DESCRIPTOR_VERSION = 2

# One kernel parameter as the emitter writes it: a buffer with its index, or
# a Metal builtin such as `uint3 _pid [[thread_position_in_grid]]`.
_PARAMETER = re.compile(
    r"^(?P<type>.+?)\s+(?P<name>\w+)\s+\[\[(?P<attr>\w+)(?:\((?P<index>\d+)\))?\]\]$"
)


def split_kernel_source(msl_source: str) -> tuple[str, list[str], str]:
    """Split emitted MSL into (header, parameter lines, body): everything
    before the kernel, the raw parameter declarations, and the kernel's
    statements without braces."""
    kernel = msl_source.find("kernel void ")
    if kernel < 0:
        raise ValueError("source is not a Palladium MSL kernel")
    open_paren = msl_source.index("(", kernel)
    close_paren = msl_source.index(")\n{", open_paren)
    close_brace = msl_source.rstrip().rfind("}")
    header = msl_source[:kernel]
    params = [
        line.strip().rstrip(",")
        for line in msl_source[open_paren + 1 : close_paren].splitlines()
        if line.strip()
    ]
    body = msl_source[close_paren + len(")\n{") : close_brace].strip("\n")
    return header, params, body


def kernel_prologue(params: list[str]) -> str:
    """Bind the emitter's parameter names inside an MLX custom kernel.

    MLX declares buffers as `arg<N>_base` in operand-then-result order and
    exposes Metal builtins under their attribute names; each emitted
    parameter becomes one declaration. Buffer names are taken from the
    source, not assumed (the attention lowering names its buffers).
    """
    lines = []
    for param in params:
        match = _PARAMETER.match(param)
        if match is None:
            raise ValueError(f"unrecognized kernel parameter {param!r}")
        ctype, name, attr = match["type"], match["name"], match["attr"]
        if attr == "buffer":
            lines.append(f"{ctype} {name} = ({ctype})arg{match['index']}_base;")
        else:
            lines.append(f"{ctype} {name} = {ctype}({attr});")
    return "\n".join(lines)


def _layout(rank: int) -> tuple[int, ...]:
    """JAX/XLA layout for a C-contiguous array (minor-to-major)."""
    return tuple(range(rank - 1, -1, -1))


def _aval_to_ir_type(aval):
    """Construct a ranked tensor type without ``aval_to_ir_type``'s unstable API."""
    return ir.RankedTensorType.get(aval.shape, mlir.dtype_to_ir_type(aval.dtype))


@dataclasses.dataclass(frozen=True)
class MpsDispatchDescriptor:
    """Static ABI sent from the JAX lowering to jax-mps. No process-local
    handles, so it can live in ``backend_config`` and be cached by PJRT.

    The handler concatenates ``header``, MLX's generated signature,
    ``prologue`` (bindings of the emitter's parameter names), and ``body``.
    ``grid`` is in threads; cooperative kernels carry the scaled grid and
    their required ``threadgroup``. Operand and result shapes come from the
    custom call's types.
    """

    version: int
    header: str
    prologue: str
    body: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int] | None
    math_mode: int

    @classmethod
    def from_spec(
        cls,
        spec: KernelSpec,
        msl_source: str,
        *,
        threadgroup: tuple[int, ...] | None,
        math_mode: int,
    ) -> MpsDispatchDescriptor:
        header, params, body = split_kernel_source(msl_source)
        grid, group = launch_geometry(spec, msl_source, threadgroup)
        return cls(
            version=_DESCRIPTOR_VERSION,
            header=header,
            prologue=kernel_prologue(params),
            body=body,
            grid=grid,
            threadgroup=group,
            math_mode=math_mode,
        )

    def to_json(self) -> str:
        """The stable, language-neutral custom-call payload."""
        payload = dataclasses.asdict(self)
        # LLVM JSON distinguishes a missing field from null: absence means the
        # handler's default launch policy, an array is an explicit size.
        if payload["threadgroup"] is None:
            del payload["threadgroup"]
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def lower_dispatch(ctx, *args, descriptor: MpsDispatchDescriptor):
    """Emit the `palladium.dispatch` custom call for a lowering context."""
    result_types = [_aval_to_ir_type(aval) for aval in ctx.avals_out]
    operand_layouts = [_layout(len(aval.shape)) for aval in ctx.avals_in]
    result_layouts = [_layout(len(aval.shape)) for aval in ctx.avals_out]
    op = mlir.custom_call(
        _CUSTOM_CALL_TARGET,
        result_types=result_types,
        operands=args,
        backend_config=descriptor.to_json(),
        api_version=2,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    return op.results


_FAST_ORDINAL = 2


def _palladium_lowering(ctx: mlir.LoweringRuleContext, *in_nodes, interpret: Any, **params):
    """Lower one pallas_call as a Palladium Metal kernel for jax-mps."""
    options = params.get("compiler_params")
    if options is None:
        options = CompilerParams()
    elif not isinstance(options, CompilerParams):
        raise TypeError(
            f"pallas_call on mps lowers through Palladium, which takes "
            f"palladium.CompilerParams, not {type(options).__name__}"
        )
    if options.math_mode != MathMode.FAST:
        raise ValueError(
            "mps lowering supports math_mode=FAST only; jax-mps's metal_kernel "
            "API does not expose Palladium's SAFE/RELAXED modes"
        )
    spec = spec_from_params(params)
    if spec.aliases:
        raise ValueError(
            "input_output_aliases are not supported on the mps path: jax-mps's "
            "MLX custom-kernel path allocates functional outputs"
        )
    check_threadgroup(spec, options.threadgroup)
    msl = emit_msl(spec, dot_general=options.dot_general)
    descriptor = MpsDispatchDescriptor.from_spec(
        spec, msl, threadgroup=options.threadgroup, math_mode=_FAST_ORDINAL
    )
    return lower_dispatch(ctx, *in_nodes, descriptor=descriptor)


def _install() -> None:
    """Route pallas_call lowering for the ``mps`` platform through Palladium.

    Registered as the primitive's common rule wrapping JAX's own, since JAX
    accepts platform-specific registrations only for platforms it already
    knows. Other platforms and ``interpret=True`` are delegated unchanged.
    """
    original = mlir._lowerings[pallas_call_p].rule

    def lowering(ctx, *in_nodes, interpret, **params):
        if interpret:
            return original(ctx, *in_nodes, interpret=interpret, **params)
        return mlir.lower_per_platform(
            ctx,
            "pallas_call",
            {"mps": _palladium_lowering},
            original,
            jax_effects.no_effects,
            *in_nodes,
            interpret=interpret,
            **params,
        )

    mlir.register_lowering(pallas_call_p, lowering)


_install()
