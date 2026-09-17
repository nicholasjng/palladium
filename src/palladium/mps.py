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
import json
import threading
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from jax._src import core as jax_core
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir

from palladium.ffi import FfiCallable
from palladium.trace import KernelSpec
from palladium.verify import DEFAULT_ATOL, DEFAULT_RTOL, verify_against

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
_DESCRIPTOR_VERSION = 1


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
    in StableHLO's ``backend_config`` and be cached by the PJRT compiler.  MSL
    is kept as text for the first implementation; a later registry can replace
    it with a content-addressed source key without changing the other fields.
    """

    version: int
    msl_source: str
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
        grid = tuple(int(d) for d in spec.grid)
        grid3 = (grid + (1, 1, 1))[:3]
        tg3 = None
        if threadgroup is not None:
            tg3 = (tuple(int(d) for d in threadgroup) + (1, 1, 1))[:3]
        return cls(
            version=_DESCRIPTOR_VERSION,
            msl_source=msl_source,
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
        for shape, dtype in zip(
            descriptor.output_shapes, descriptor.output_dtypes, strict=True
        )
    )


_mps_dispatch_p.def_abstract_eval(_abstract_eval)
_mps_dispatch_p.def_impl(lambda *args, fallback, **_: _as_tuple(fallback(*args)))


def _fallback_lowering(ctx, *args, fallback, **_):
    # A portable fallback is essential: callers can retain one JAX program
    # across CPU, CUDA, and mps, and it makes the custom call testable before
    # jax-mps is present.
    return mlir.lower_fun(lambda *xs: _as_tuple(fallback(*xs)), multiple_results=True)(
        ctx, *args
    )


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


class MpsCallable:
    """A Pallas kernel that becomes ``palladium.dispatch`` on jax-mps.

    This class intentionally reuses :class:`FfiCallable`'s tracing, MSL
    emission, diagnostics, and bounded shape cache.  It does *not* use its
    CPU FFI dispatch: MPS buffers belong to jax-mps, so only the descriptor
    crosses this boundary.
    """

    def __init__(
        self,
        kernel: Callable,
        pallas_kwargs: dict[str, Any],
        math_mode: Any,
        threadgroup: int | tuple[int, ...] | None,
        cache_size: int,
    ) -> None:
        # FfiCallable owns the well-tested trace/emit cache.  Its public call
        # path is never invoked here.
        self._staged = FfiCallable(
            kernel,
            pallas_kwargs,
            math_mode,
            vmap_method=None,
            threadgroup=threadgroup,
            cache_size=cache_size,
        )
        self.interpret = self._staged.interpret

    @property
    def _cache(self):
        """Expose the specialization cache for diagnostics and tests."""
        return self._staged._cache

    def explain(self, *args):
        return self._staged.explain(*args)

    def verify(
        self,
        *args,
        reference=None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
    ):
        """Run the selected platform path and compare it with a reference."""
        shapes = [jax.ShapeDtypeStruct(a.shape, a.dtype) for a in args]
        outputs = verify_against(
            self,
            None if reference is not None else self.interpret,
            self._staged._spec_and_msl(tuple(shapes))[0].uses_threadgroup,
            args,
            reference,
            rtol,
            atol,
        )
        return outputs[0] if len(outputs) == 1 else outputs

    def with_reference_vjp(self, reference: Callable) -> Callable:
        """Attach a correctness-first VJP implemented by a JAX reference.

        The primal executes this callable, and therefore remains one MPS
        custom call on jax-mps.  The pullback is calculated by JAX from
        ``reference``.  This is useful for bringing a fused forward solve into
        an end-to-end training loop while a Pallas discrete-adjoint kernel is
        being developed and validated.

        ``reference`` must have the same inputs, outputs, and differentiable
        semantics as this call.  Its VJP is intentionally *not* a performance
        solution: a training-speed claim requires a native backward kernel.
        """

        @jax.custom_vjp
        def differentiated(*args):
            return self(*args)

        def forward(*args):
            return self(*args), args

        def backward(residual, cotangents):
            _, pullback = jax.vjp(reference, *residual)
            return pullback(cotangents)

        differentiated.defvjp(forward, backward)
        return differentiated

    def __call__(self, *args):
        _register_mps_lowering()
        spec, msl_source = self._staged._spec_and_msl(args)
        if spec.aliases:
            raise ValueError(
                "mps_call_jit does not yet support input_output_aliases: "
                "jax-mps's MLX custom-kernel path allocates functional outputs"
            )
        if self._staged._math_mode != 2:  # metal_runtime's FAST ordinal
            raise ValueError(
                "mps_call_jit currently supports math_mode=FAST only; "
                "jax-mps's metal_kernel API does not expose Palladium's "
                "SAFE/RELAXED compilation modes"
            )
        descriptor = MpsDispatchDescriptor.from_spec(
            spec,
            msl_source,
            threadgroup=self._staged._threadgroup,
            math_mode=self._staged._math_mode,
        )
        outputs = _mps_dispatch_p.bind(
            *args, descriptor=descriptor, fallback=self.interpret
        )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def mps_call_jit(
    kernel: Callable, *, vjp_reference: Callable | None = None, **pallas_kwargs
) -> MpsCallable | Callable:
    """Create a Pallas call that lowers to a jax-mps Metal custom call.

    On the ``mps`` platform this emits ``stablehlo.custom_call
    @palladium.dispatch``.  Other platforms execute Pallas's interpreter as a
    portable fallback.  The jax-mps native handler is responsible for zero-copy
    buffer wrapping and command-stream ordering.

    ``vmap`` is deliberately unsupported in v1; place an independent batch
    dimension directly in the Pallas grid so one invocation is one dispatch.
    Pass ``vjp_reference`` to opt into a correctness-first custom VJP: the
    forward pass is the MPS custom call and the backward pass is generated from
    the matching pure-JAX reference.  A Pallas discrete-adjoint kernel remains
    necessary for fused training performance.
    """
    from metal_runtime import MathMode

    math_mode = pallas_kwargs.pop("math_mode", MathMode.FAST)
    threadgroup = pallas_kwargs.pop("threadgroup", None)
    cache_size = pallas_kwargs.pop("cache_size", 256)
    vmap_method = pallas_kwargs.pop("vmap_method", None)
    if vmap_method is not None:
        raise ValueError(
            "mps_call_jit does not batch a custom call in v1; put the batch "
            "dimension in the Pallas grid so the whole batch is one dispatch"
        )
    call = MpsCallable(kernel, pallas_kwargs, math_mode, threadgroup, cache_size)
    return call if vjp_reference is None else call.with_reference_vjp(vjp_reference)
