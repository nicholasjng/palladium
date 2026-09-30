"""Shared TensorOps descriptors, views, and jaxpr utilities."""

from __future__ import annotations

import dataclasses

from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var, subjaxprs

from palladium.emit.core import Cursor, CVal
from palladium.errors import EmitError
from palladium.trace import BlockInfo, KernelSpec

SIMDGROUPS = 4


def _kernel_source(name: str, parameters: tuple[str, ...], body: list[str]) -> str:
    """Wrap emitted statements in the common Metal TensorOps kernel preamble."""
    parameter_text = ",\n    ".join(parameters)
    return "\n".join(
        (
            "#include <metal_stdlib>",
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>",
            "",
            "using namespace metal;",
            "using namespace mpp;",
            "",
            f"kernel void {name}(\n    {parameter_text})\n{{",
            *body,
            "}",
            "",
        )
    )


def _shape(atom: Var | Literal) -> tuple[int, ...]:
    return tuple(int(size) for size in getattr(atom.aval, "shape", ()))


@dataclasses.dataclass(frozen=True)
class _TensorHandle:
    """MPP tensor expression emitted from a typed backing-buffer view."""

    expr: str
    shape: tuple[int, ...]
    ctype: str
    space: str


@dataclasses.dataclass(frozen=True)
class _TensorView:
    """Typed MPP tensor view over an emitter value and explicit strides."""

    storage: CVal
    shape: tuple[int, ...]
    extents: tuple[str | int, ...]
    strides: tuple[str | int, ...]

    def __post_init__(self) -> None:
        if len(self.extents) != len(self.strides) or len(self.shape) != len(self.extents):
            raise ValueError("TensorOps shape, extents, and strides must have the same rank")
        if self.storage.space not in ("device", "threadgroup"):
            raise ValueError("TensorOps storage must be device or threadgroup memory")

    def emit(self, cursor: Cursor, name: str) -> _TensorHandle:
        """Declare an MPP tensor handle over this typed storage view."""
        rank = len(self.extents)
        extents = ", ".join(map(str, self.extents))
        strides = ", ".join(map(str, self.strides))
        cursor.emit(
            f"auto {name} = tensor<{self.storage.space} {self.storage.ctype}, "
            f"dextents<int, {rank}>, tensor_inline>("
        )
        cursor.indent += 1
        cursor.emit(f"{self.storage.expr}, dextents<int, {rank}>({extents}),")
        cursor.emit(f"array<int, {rank}>{{{strides}}});")
        cursor.indent -= 1
        return _TensorHandle(name, self.shape, self.storage.ctype, self.storage.space)


@dataclasses.dataclass(frozen=True)
class _TensorOpsMatmul:
    """TensorOps descriptor inferred from one unbatched jaxpr dot."""

    name: str
    descriptor: str
    m: int
    n: int
    k: int
    transpose_lhs: bool
    transpose_rhs: bool
    accumulate: bool

    @classmethod
    def from_eqn(
        cls,
        eqn: JaxprEqn,
        producers: dict[Var, JaxprEqn],
        *,
        name: str,
        accumulate: bool,
    ) -> _TensorOpsMatmul:
        """Recognize an ordinary rank-2 contraction and its lazy transposes."""
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params["dimension_numbers"]
        if lhs_batch or rhs_batch or tuple(lhs_contract) != (1,) or tuple(rhs_contract) != (0,):
            raise EmitError("TensorOps dot requires an unbatched row-major matrix contraction")
        lhs_shape = _shape(eqn.invars[0])
        rhs_shape = _shape(eqn.invars[1])
        out_shape = _shape(eqn.outvars[0])
        if len(lhs_shape) != 2 or len(rhs_shape) != 2 or len(out_shape) != 2:
            raise EmitError("TensorOps dot requires rank-2 operands and result")
        m, k = lhs_shape
        kr, n = rhs_shape
        if k != kr or out_shape != (m, n):
            raise EmitError("TensorOps dot dimensions do not match")

        def is_transposed(atom) -> bool:
            producer = producers.get(atom) if isinstance(atom, Var) else None
            return (
                producer is not None
                and producer.primitive.name == "transpose"
                and tuple(producer.params["permutation"]) == (1, 0)
            )

        return cls(
            name=name,
            descriptor=f"{name}_desc",
            m=m,
            n=n,
            k=k,
            transpose_lhs=is_transposed(eqn.invars[0]),
            transpose_rhs=is_transposed(eqn.invars[1]),
            accumulate=accumulate,
        )

    def emit_declaration(self, cursor: Cursor) -> None:
        """Emit this dot's MPP descriptor and cooperative operation handle."""
        trans_a = str(self.transpose_lhs).lower()
        trans_b = str(self.transpose_rhs).lower()
        mode = (
            ", tensor_ops::matmul2d_descriptor::mode::multiply_accumulate"
            if self.accumulate
            else ""
        )
        cursor.emit(
            f"constexpr tensor_ops::matmul2d_descriptor {self.descriptor}({self.m}, {self.n}, "
            f"{self.k}, {trans_a}, {trans_b}, false{mode});"
        )
        cursor.emit(
            f"tensor_ops::matmul2d<{self.descriptor}, execution_simdgroups<{SIMDGROUPS}>> "
            f"{self.name};"
        )

    def emit_run(
        self, cursor: Cursor, lhs: _TensorHandle, rhs: _TensorHandle, out: _TensorHandle
    ) -> None:
        """Emit the operation call for typed tensor handles."""
        cursor.emit(f"{self.name}.run({lhs.expr}, {rhs.expr}, {out.expr});")

    def emit_cooperative_destination(
        self,
        cursor: Cursor,
        lhs: _TensorHandle,
        rhs: _TensorHandle,
        name: str = "cTc",
        element_type: str = "float",
    ) -> _TensorHandle:
        """Create the MPP-owned destination layout used for fused epilogues."""
        cursor.emit(
            f"auto {name} = {self.name}.get_destination_cooperative_tensor<"
            f"decltype({lhs.expr}), decltype({rhs.expr}), {element_type}>();"
        )
        return _TensorHandle(name, (self.m, self.n), element_type, "thread")


def _index_map_is(info: BlockInfo, axes: tuple[int | None, ...]) -> bool:
    """Whether the map returns the named grid axes or integer literals."""
    jaxpr = info.index_map_jaxpr.jaxpr
    if jaxpr.eqns or len(jaxpr.invars) < max((a for a in axes if a is not None), default=-1) + 1:
        return False
    if len(jaxpr.outvars) != len(axes):
        return False
    for outvar, axis in zip(jaxpr.outvars, axes, strict=True):
        if axis is None:
            if not isinstance(outvar, Literal) or int(outvar.val) != 0:
                return False
        elif outvar is not jaxpr.invars[axis]:
            return False
    return True


def _contains(jaxpr: Jaxpr, primitive: str) -> bool:
    """Whether a jaxpr stages `primitive`, including in nested control flow."""
    return any(eqn.primitive.name == primitive for eqn in jaxpr.eqns) or any(
        _contains(child, primitive) for child in subjaxprs(jaxpr)
    )


def emits_cooperative(msl_source: str) -> bool:
    """Whether emitted MSL runs one threadgroup per Pallas program.

    Only the TensorOps lowerings address programs by threadgroup position,
    so the source itself, not the selection policy, decides the launch
    geometry.
    """
    return "threadgroup_position_in_grid" in msl_source


def cooperative_launch(
    grid: tuple[int, ...], simd_width: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """The (threadgroup, grid) Metal launch for a cooperative kernel.

    One threadgroup of `SIMDGROUPS` SIMD groups per program, so the thread
    grid is the Pallas grid scaled by the threadgroup along x.
    """
    threadgroup = (simd_width * SIMDGROUPS, 1, 1)
    padded = (tuple(int(g) for g in grid) + (1, 1, 1))[:3]
    scaled = tuple(g * t for g, t in zip(padded, threadgroup, strict=True))
    return threadgroup, (scaled[0], scaled[1], scaled[2])


def uses_tensorops(spec: KernelSpec, dot_general: str) -> bool:
    """Whether this kernel's requested policy selects cooperative TensorOps.

    "tensorops" requires it, "default" never selects it, and "auto" selects
    it for dots on a 2D or 3D grid.
    """
    if not _contains(spec.jaxpr, "dot_general"):
        return False
    return dot_general == "tensorops" or (dot_general == "auto" and len(spec.grid) in (2, 3))
