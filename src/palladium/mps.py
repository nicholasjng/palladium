"""The ``palladium.dispatch`` custom-call ABI shared with jax-mps.

On the ``mps`` platform the pallas_call lowering emits a StableHLO custom
call whose ``backend_config`` is ``MpsDispatchDescriptor.to_json()``;
jax-mps's handler for ``MPS_CUSTOM_CALL_TARGET`` builds an MLX kernel from
it on its own Metal stream. Nothing here executes anything.
"""

from __future__ import annotations

import dataclasses
import json
import re

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


# An ordinary StableHLO custom-call target, not a jax.ffi target: jax-mps
# owns the buffers and encodes the dispatch on its own Metal stream.
MPS_CUSTOM_CALL_TARGET = "palladium.dispatch"
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
        grid = tuple(int(d) for d in spec.grid)
        grid3 = (grid + (1, 1, 1))[:3]
        tg3 = None
        if threadgroup is not None:
            tg3 = (tuple(int(d) for d in threadgroup) + (1, 1, 1))[:3]
        if emits_cooperative(msl_source):
            # One threadgroup per program; the thread grid scales to match.
            required, grid3 = cooperative_launch(grid, simdgroup_width())
            if tg3 is not None and tg3 != required:
                raise ValueError(f"cooperative kernel requires threadgroup={required}, got {tg3}")
            tg3 = required
        return cls(
            version=_DESCRIPTOR_VERSION,
            header=header,
            prologue=kernel_prologue(params),
            body=body,
            grid=grid3,
            threadgroup=tg3,
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

    @classmethod
    def from_json(cls, value: str) -> MpsDispatchDescriptor:
        """Parse and validate a descriptor."""
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
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    return op.results
