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

from palladium.emit import emit_msl
from palladium.emit.core import shaped
from palladium.launch import CompilerParams, MathMode, check_threadgroup, launch_geometry
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
    """Separate a Palladium kernel into its header, parameter declarations, and body.

    Parameters
    ----------
    msl_source : str
        Complete source using the signature and brace formatting produced by Palladium.

    Returns
    -------
    header : str
        Text before ``kernel void``.
    params : list of str
        Stripped parameter declarations without trailing commas.
    body : str
        Statements without the signature or outer braces.

    Raises
    ------
    ValueError
        The kernel marker or expected signature delimiter is missing.

    Notes
    -----
    This is a parser for the emitter format, not arbitrary MSL. It assumes one kernel with
    the final closing brace at the end of the source.
    """
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
    """Bind emitter parameters to MLX custom-kernel buffers and Metal builtins.

    Parameters
    ----------
    params : list of str
        Parameter declarations returned by ``split_kernel_source``.

    Returns
    -------
    str
        MSL declarations binding buffers by index and builtins by attribute name; empty
        for no parameters.

    Raises
    ------
    ValueError
        A declaration does not match the expected parameter format.

    Notes
    -----
    MLX buffers are named ``arg<N>_base`` in input-then-output order. Names in the original
    source need not follow the arg<N> convention.
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


@dataclasses.dataclass(frozen=True)
class MpsDispatchDescriptor:
    """Carry a serializable Metal dispatch recipe from JAX lowering to jax-mps.

    Attributes
    ----------
    version : int
        Descriptor ABI version; ``from_spec`` currently produces version 2.
    header : str
        Includes, namespaces, and helper functions preceding the kernel.
    prologue : str
        Bindings from MLX argument bases and builtins to emitter parameter names.
    body : str
        Kernel statements without the signature or outer braces.
    grid : tuple of int
        Three-dimensional thread grid, including cooperative group scaling.
    threadgroup : tuple of int or None
        Explicit group dimensions or None for the handler default.
    math_mode : int
        Native ABI math-mode ordinal; the current mps lowering supplies FAST (2).

    Notes
    -----
    The JSON payload contains no process-local handles and can be cached in StableHLO
    backend_config. Shapes and layouts travel in custom-call types. The dataclass does not
    itself validate backend option support.
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
        """Build a dispatch descriptor from emitted source and launch options.

        Parameters
        ----------
        spec : KernelSpec
            Traced Pallas kernel, including operand layouts, scratch buffers, and grid.
        msl_source : str
            Complete source produced by a Palladium emitter.
        threadgroup : tuple of int or None, optional
            One to three positive dimensions, or None for runtime selection. Cooperative
            kernels require their fixed group size.
        math_mode : int
            ABI math-mode ordinal, stored verbatim; the mps lowering separately enforces
            FAST.

        Returns
        -------
        MpsDispatchDescriptor
            Versioned source fragments, parameter bindings, and resolved geometry.

        Raises
        ------
        ValueError
            Source is not in the expected emitter format, parameters are unrecognized, or a
            cooperative group does not match.

        Notes
        -----
        This splits source without compiling it. Buffer bindings use emitted buffer indices, so
        named attention buffers and primitive arg buffers share the same ABI.
        """
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
        """Serialize the descriptor as a deterministic custom-call payload.

        Returns
        -------
        str
            Compact JSON with sorted keys. An automatic threadgroup is omitted rather than
            encoded as null.
        """
        payload = dataclasses.asdict(self)
        # LLVM JSON distinguishes a missing field from null: absence means the
        # handler's default launch policy, an array is an explicit size.
        if payload["threadgroup"] is None:
            del payload["threadgroup"]
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


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
    check_threadgroup(spec, options.threadgroup_dims)
    msl = emit_msl(spec, dot_general=options.dot_general, program=options.program)
    descriptor = MpsDispatchDescriptor.from_spec(
        spec, msl, threadgroup=options.threadgroup_dims, math_mode=_FAST_ORDINAL
    )
    # The jax-mps palladium.dispatch handler reads the descriptor. Result types
    # are built directly: mlir.aval_to_ir_type's signature is not stable.
    return mlir.custom_call(
        _CUSTOM_CALL_TARGET,
        result_types=[
            ir.RankedTensorType.get(aval.shape, mlir.dtype_to_ir_type(aval.dtype))
            for aval in map(shaped, ctx.avals_out)
        ],
        operands=in_nodes,
        backend_config=descriptor.to_json(),
        api_version=2,
        operand_layouts=[_layout(len(aval.shape)) for aval in map(shaped, ctx.avals_in)],
        result_layouts=[_layout(len(aval.shape)) for aval in map(shaped, ctx.avals_out)],
    ).results


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
