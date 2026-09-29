"""The ``palladium.dispatch`` custom-call ABI shared with jax-mps.

Palladium traces a Pallas kernel and emits MSL; on the ``mps`` platform the
pallas_call lowering (``palladium.pallas_backend``) turns that into a
StableHLO custom call whose ``backend_config`` is
``MpsDispatchDescriptor.to_json()``. jax-mps's handler for
``MPS_CUSTOM_CALL_TARGET`` builds an MLX kernel from it on its own Metal
stream. Nothing here executes anything.
"""

from __future__ import annotations

import dataclasses
import json
import re

import numpy as np
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

from palladium.diagnostics import simdgroup_width
from palladium.emit.tensorops import cooperative_launch, emits_cooperative
from palladium.trace import KernelSpec

__all__ = [
    "MPS_CUSTOM_CALL_TARGET",
    "MpsDispatchDescriptor",
    "kernel_prologue",
    "lower_dispatch",
    "split_kernel_source",
]


# This is intentionally an ordinary StableHLO custom-call target, rather than
# a jax.ffi target: jax-mps owns MPS buffers and must encode the dispatch on
# its existing Metal stream.
MPS_CUSTOM_CALL_TARGET = "palladium.dispatch"
_DESCRIPTOR_VERSION = 2

# One kernel parameter as the emitter writes it: a buffer with its index, or
# a Metal builtin such as `uint3 _pid [[thread_position_in_grid]]`.
_PARAMETER = re.compile(
    r"^(?P<type>.+?)\s+(?P<name>\w+)\s+\[\[(?P<attr>\w+)(?:\((?P<index>\d+)\))?\]\]$"
)


def split_kernel_source(msl_source: str) -> tuple[str, list[str], str]:
    """Split emitted MSL into (header, parameter lines, body).

    The header is everything before the kernel: includes, using directives,
    and helper functions. Parameters are the raw parameter declarations. The
    body is the kernel's statement list without its braces.
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
    """Bind the emitter's parameter names inside an MLX custom kernel.

    MLX declares buffers itself, named `arg<N>_base` by the handler in
    operand-then-result order, and exposes Metal builtins under their
    attribute names. Each emitted parameter becomes one declaration: buffer
    N cast to the emitter's own qualifier and name, builtins constructed
    from the attribute. The emitter's buffer names are not assumed: the
    attention lowering calls its buffers query, key, value, and output.
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
    """Static ABI sent from the JAX lowering to jax-mps.

    The descriptor contains no process-local handles.  It can therefore live
    in StableHLO's ``backend_config`` and be cached by the PJRT compiler.

    The kernel travels as three pieces of text the handler concatenates with
    MLX's generated signature between them: ``header`` (includes, using
    directives, helper functions), ``prologue`` (declarations binding the
    emitter's parameter names to MLX's buffers and builtins), and ``body``.
    ``grid`` is in threads; cooperative kernels carry the scaled grid and
    their required ``threadgroup``.
    """

    version: int
    header: str
    prologue: str
    body: str
    function_name: str
    grid: tuple[int, int, int]
    threadgroup: tuple[int, int, int] | None
    math_mode: int
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    aliases: tuple[tuple[int, int], ...]

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
        grid = tuple(int(d) for d in spec.grid)
        grid3 = (grid + (1, 1, 1))[:3]
        tg3 = None
        if threadgroup is not None:
            tg3 = (tuple(int(d) for d in threadgroup) + (1, 1, 1))[:3]
        if emits_cooperative(msl_source):
            # One threadgroup per program: the source addresses programs by
            # threadgroup position, so the thread grid is scaled to match.
            required, grid3 = cooperative_launch(grid, simdgroup_width())
            if tg3 is not None and tg3 != required:
                raise ValueError(f"cooperative kernel requires threadgroup={required}, got {tg3}")
            tg3 = required
        return cls(
            version=_DESCRIPTOR_VERSION,
            header=header,
            prologue=kernel_prologue(params),
            body=body,
            function_name=spec.name,
            grid=grid3,
            threadgroup=tg3,
            math_mode=math_mode,
            input_shapes=tuple(tuple(info.array_shape) for info in spec.inputs),
            input_dtypes=tuple(np.dtype(info.dtype).str for info in spec.inputs),
            output_shapes=tuple(tuple(info.array_shape) for info in spec.outputs),
            output_dtypes=tuple(np.dtype(info.dtype).str for info in spec.outputs),
            aliases=spec.aliases,
        )

    def to_json(self) -> str:
        """The stable, language-neutral custom-call payload."""
        payload = dataclasses.asdict(self)
        # LLVM JSON distinguishes a missing field from a present null. The MPS
        # handler uses absence to request its ordinary independent-thread
        # launch policy, while a present array is an explicit cooperative size.
        if payload["threadgroup"] is None:
            del payload["threadgroup"]
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> MpsDispatchDescriptor:
        """Parse and validate a descriptor in tests or native-adapter shims."""
        raw = json.loads(value)
        if raw.get("version") != _DESCRIPTOR_VERSION:
            raise ValueError(
                f"unsupported Palladium MPS descriptor version {raw.get('version')!r}; "
                f"expected {_DESCRIPTOR_VERSION}"
            )
        for name in ("grid", "threadgroup"):
            if raw.get(name) is not None:
                raw[name] = tuple(raw[name])
        raw.setdefault("threadgroup", None)
        for name in ("input_shapes", "output_shapes", "aliases"):
            raw[name] = tuple(tuple(item) for item in raw[name])
        raw["input_dtypes"] = tuple(raw["input_dtypes"])
        raw["output_dtypes"] = tuple(raw["output_dtypes"])
        return cls(**raw)


def lower_dispatch(ctx, *args, descriptor: MpsDispatchDescriptor):
    """Emit the `palladium.dispatch` custom call for a lowering context."""
    result_types = [_aval_to_ir_type(aval) for aval in ctx.avals_out]
    operand_layouts = [_layout(len(aval.shape)) for aval in ctx.avals_in]
    result_layouts = [_layout(len(aval.shape)) for aval in ctx.avals_out]
    op = mlir.custom_call(
        MPS_CUSTOM_CALL_TARGET,
        result_types=result_types,
        operands=args,
        backend_config=descriptor.to_json(),
        api_version=2,
        operand_output_aliases=dict(descriptor.aliases) or None,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    return op.results
