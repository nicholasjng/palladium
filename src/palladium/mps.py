"""JAX ``mps`` custom-call bridge for Palladium Pallas kernels.

This is deliberately a narrow integration point.  Palladium still traces a
user-authored Pallas kernel and emits MSL; on the ``mps`` platform the result
is a StableHLO custom call for jax-mps to encode on its Metal stream.  Other
platforms lower to Pallas's interpreter, which is both a portable fallback
and the semantic reference while the jax-mps handler is developed.

The native counterpart is expected to handle ``MPS_CUSTOM_CALL_TARGET`` and
consume ``MpsDispatchDescriptor.to_json()`` as its backend_config.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import re
import threading
from collections.abc import Callable
from typing import Any, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core as jax_core, dispatch as jax_dispatch
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

from palladium._callable import CallOptions, PallasCallable
from palladium.diagnostics import simdgroup_width
from palladium.emit.tensorops import cooperative_launch, emits_cooperative
from palladium.ffi import _MATH_MODE_ORDINALS
from palladium.trace import KernelSpec

__all__ = [
    "MPS_CUSTOM_CALL_TARGET",
    "MpsCallable",
    "MpsDispatchDescriptor",
    "mps_call_jit",
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


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


_mps_dispatch_p = jax_core.Primitive("palladium_mps_dispatch")
_mps_dispatch_p.multiple_results = True


def _abstract_eval(*_, descriptor: MpsDispatchDescriptor, **__) -> tuple[Any, ...]:
    return tuple(
        jax_core.ShapedArray(shape, np.dtype(dtype))
        for shape, dtype in zip(descriptor.output_shapes, descriptor.output_dtypes, strict=True)
    )


_mps_dispatch_p.def_abstract_eval(_abstract_eval)
_mps_dispatch_p.def_impl(functools.partial(jax_dispatch.apply_primitive, _mps_dispatch_p))


def _batching(batched_args, batch_dims, **params):
    """`jax.vmap` over the custom call: one dispatch per batch element.

    The launch grid is baked into the descriptor for the unbatched shape,
    so the batch cannot be folded into it after tracing; `lax.map` runs the
    elements sequentially instead. A batch axis in the Pallas grid is one
    dispatch total and is the faster choice when the kernel allows it.
    """
    # Batch dims are ints for mapped operands and None for unmapped ones.
    mapped = [d is not None for d in batch_dims]
    size = next(a.shape[d] for a, d in zip(batched_args, batch_dims, strict=True) if d is not None)
    moved = [
        a if d is None else jnp.moveaxis(a, d, 0)
        for a, d in zip(batched_args, batch_dims, strict=True)
    ]

    def element(index):
        args = [a[index] if m else a for a, m in zip(moved, mapped, strict=True)]
        return tuple(_mps_dispatch_p.bind(*args, **params))

    outs = jax.lax.map(element, jnp.arange(size))
    return tuple(outs), (0,) * len(outs)


def _register_batching() -> None:
    from jax._src.interpreters import batching

    batching.primitive_batchers[_mps_dispatch_p] = _batching


_register_batching()


def _fallback_lowering(ctx, *args, fallback, allow_fallback=True, cooperative=False, **_):
    if not allow_fallback:
        raise ValueError(
            "MPS backend required (fallback='error'); this computation is being lowered for another platform"
        )
    if cooperative:
        raise ValueError(
            "cooperative kernel cannot use the Pallas interpreter fallback: "
            "it models threadgroups of one; select MPS or use metal_call_jit"
        )
    # A portable fallback is essential: callers can retain one JAX program
    # across CPU, CUDA, and mps, and it makes the custom call testable before
    # jax-mps is present.
    return mlir.lower_fun(lambda *xs: _as_tuple(fallback(*xs)), multiple_results=True)(ctx, *args)


def _mps_lowering(ctx, *args, descriptor: MpsDispatchDescriptor, **_):
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


mlir.register_lowering(_mps_dispatch_p, _fallback_lowering)
_MPS_LOWERING_LOCK = threading.Lock()
_mps_lowering_registered = False


def _register_mps_lowering() -> None:
    """Register after jax-mps has made ``mps`` a known JAX platform.

    Importing Palladium on a CPU-only installation must remain harmless: JAX
    refuses a platform-specific rule for an undiscovered plugin.  A call is
    traced only after its backend has been selected, which is the right point
    to install this rule.
    """
    global _mps_lowering_registered
    if _mps_lowering_registered:
        return
    with _MPS_LOWERING_LOCK:
        if _mps_lowering_registered:
            return
        try:
            mlir.register_lowering(_mps_dispatch_p, _mps_lowering, platform="mps")
        except NotImplementedError:
            # No jax-mps plugin is installed/initialized. The generic CPU
            # fallback remains registered and is intentionally usable.
            return
        _mps_lowering_registered = True


class MpsCallable(PallasCallable):
    """A Pallas kernel that becomes ``palladium.dispatch`` on jax-mps.

    Shares tracing, MSL emission, diagnostics, and the per-shape cache with
    the other paths; only the descriptor crosses to jax-mps, which owns the
    MPS buffers.
    """

    execution_path = "mps-or-pallas-interpret (selected at lowering)"

    def __init__(self, kernel: Callable, pallas_kwargs: dict[str, Any], options: CallOptions):
        if options.fallback not in ("interpret", "error"):
            raise ValueError("fallback must be 'interpret' or 'error'")
        super().__init__(kernel, pallas_kwargs, options)
        self._allow_fallback = options.fallback == "interpret"

    def explain(self, *args, platform: str | None = None):
        """Report the expected path; explicit JIT placement can override inference.

        Pass platform= when explaining a computation intended for a specific
        JIT target. Actual fallback policy is enforced during lowering.
        """
        if platform is None:
            platforms = {
                d.platform
                for a in args
                if isinstance(a, jax.Array) and a.committed
                for d in a.devices()
            }
            device = jax.config.jax_default_device
            platform = (
                next(iter(platforms))
                if len(platforms) == 1
                else device.platform
                if device is not None
                else jax.default_backend()
            )
        diagnostics = super().explain(*args)
        path = "mps-custom-call" if platform == "mps" else f"pallas-interpret:{platform}"
        if platform != "mps" and (not self._allow_fallback or diagnostics.cooperative):
            path = f"rejected:{platform} (requires mps)"
        return dataclasses.replace(diagnostics, execution_path=path)

    def __call__(self, *args):
        _register_mps_lowering()
        spec, msl_source, _ = self._spec_and_msl(args)
        if spec.aliases:
            raise ValueError(
                "mps_call_jit does not yet support input_output_aliases: "
                "jax-mps's MLX custom-kernel path allocates functional outputs"
            )
        math_mode = _MATH_MODE_ORDINALS[self._options.math_mode]
        if math_mode != 2:  # metal_runtime's FAST ordinal
            raise ValueError(
                "mps_call_jit currently supports math_mode=FAST only; "
                "jax-mps's metal_kernel API does not expose Palladium's "
                "SAFE/RELAXED compilation modes"
            )
        descriptor = MpsDispatchDescriptor.from_spec(
            spec,
            msl_source,
            threadgroup=self._options.threadgroup,
            math_mode=math_mode,
        )
        outputs = _mps_dispatch_p.bind(
            *args,
            descriptor=descriptor,
            fallback=self.interpret,
            allow_fallback=self._allow_fallback,
            cooperative=spec.uses_threadgroup,
        )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


@overload
def mps_call_jit(
    kernel: Callable, *, vjp_reference: None = None, fallback: str = "interpret", **pallas_kwargs
) -> MpsCallable: ...


@overload
def mps_call_jit(
    kernel: Callable, *, vjp_reference: Callable, fallback: str = "interpret", **pallas_kwargs
) -> Callable: ...


def mps_call_jit(
    kernel: Callable,
    *,
    vjp_reference: Callable | None = None,
    fallback: str = "interpret",
    **pallas_kwargs,
) -> MpsCallable | Callable:
    """Create a Pallas call that lowers to a jax-mps Metal custom call.

    On the ``mps`` platform this emits ``stablehlo.custom_call
    @palladium.dispatch``.  Other platforms execute Pallas's interpreter as a
    portable fallback.  The jax-mps native handler is responsible for zero-copy
    buffer wrapping and command-stream ordering.

    Pass ``fallback="error"`` to require MPS at lowering time, including
    eager calls. Cooperative kernels always reject the interpreter fallback
    because it models groups of one. ``explain`` reports the expected path;
    explicit JIT placement can override its platform inference.

    ``jax.vmap`` runs one dispatch per batch element through ``lax.map``;
    a batch dimension in the Pallas grid is one dispatch in total and is
    preferable when the kernel allows it.
    Pass ``vjp_reference`` to opt into a correctness-first custom VJP: the
    forward pass is the MPS custom call and the backward pass is generated from
    the matching pure-JAX reference.  A Pallas discrete-adjoint kernel remains
    necessary for fused training performance.
    """
    options = CallOptions.split(pallas_kwargs, vmap_method="sequential", fallback=fallback)
    if options.vmap_method not in (None, "sequential"):
        raise ValueError(
            "mps_call_jit batches jax.vmap sequentially (one dispatch per "
            "element through lax.map); other vmap methods are not available "
            "because the launch grid is baked per unbatched shape. Put the "
            "batch dimension in the Pallas grid for one dispatch in total."
        )
    call = MpsCallable(kernel, pallas_kwargs, options)
    return call if vjp_reference is None else call.with_reference_vjp(vjp_reference)
