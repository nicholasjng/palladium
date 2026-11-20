"""Shared TensorOps descriptors, views, and jaxpr utilities."""

from __future__ import annotations

import dataclasses
import re

from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var, subjaxprs

from palladium.emit.core import Cursor
from palladium.errors import EmitError
from palladium.trace import BlockInfo, KernelSpec

SIMDGROUPS = 4
# Threads per SIMD group on every Apple GPU; the fragment layouts assume it.
SIMD_WIDTH = 32
UNROLL = '_Pragma("clang loop unroll(full)")'


def _kernel_source(
    name: str, parameters: tuple[str, ...], cursor: Cursor, width: int | None = None
) -> str:
    """Wrap emitted statements in the common Metal TensorOps kernel preamble.

    The body starts by declaring the group width for the launch: `width`
    threads, by default four SIMD groups.
    """
    parameter_text = ",\n    ".join(parameters)
    width = width or SIMDGROUPS * SIMD_WIDTH
    return "\n".join(
        (
            "#include <metal_stdlib>",
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>",
            "",
            "using namespace metal;",
            "using namespace mpp;",
            "",
            *cursor.helpers.values(),
            f"kernel void {name}(\n    {parameter_text})\n{{",
            f"    // palladium: threadgroup width {width}",
            *cursor.lines,
            "}",
            "",
        )
    )


def _shape(atom: Var | Literal) -> tuple[int, ...]:
    """Read static dimensions from an atom; an absent shape is treated as scalar."""
    return tuple(int(size) for size in getattr(atom.aval, "shape", ()))


def _tensor(
    cursor: Cursor,
    name: str,
    element: str,
    pointer: str,
    extents: tuple[str | int, ...],
    strides: tuple[str | int, ...],
) -> str:
    """Declare the MPP tensor `name` over `pointer` and return the name.

    `element` is the address space and type, such as "device float"; extents and
    strides are in elements, innermost dimension first.
    """
    rank = len(extents)
    cursor.emit(f"auto {name} = tensor<{element}, dextents<int, {rank}>, tensor_inline>(")
    cursor.indent += 1
    cursor.emit(f"{pointer}, dextents<int, {rank}>({', '.join(map(str, extents))}),")
    cursor.emit(f"array<int, {rank}>{{{', '.join(map(str, strides))}}});")
    cursor.indent -= 1
    return name


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

    def emit_cooperative_destination(
        self,
        cursor: Cursor,
        lhs: str,
        rhs: str,
        name: str = "cTc",
        element_type: str = "float",
    ) -> str:
        """Declare `name`, the MPP-owned destination of this product for operands
        typed like the tensors `lhs` and `rhs`, used for fused epilogues."""
        cursor.emit(
            f"auto {name} = {self.name}.get_destination_cooperative_tensor<"
            f"decltype({lhs}), decltype({rhs}), {element_type}>();"
        )
        return name


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


def _producers(jaxpr: Jaxpr) -> dict[Var, JaxprEqn]:
    """The equation defining each variable of `jaxpr`'s top level."""
    return {variable: eqn for eqn in jaxpr.eqns for variable in eqn.outvars}


def _unsqueezed(block_shape: tuple[int | None, ...]) -> tuple[int, ...]:
    """A block shape with each squeezed dimension as its one element."""
    return tuple(1 if size is None else int(size) for size in block_shape)


def _contains(jaxpr: Jaxpr, primitive: str) -> bool:
    """Whether a jaxpr stages `primitive`, including in nested control flow."""
    return any(eqn.primitive.name == primitive for eqn in jaxpr.eqns) or any(
        _contains(child, primitive) for child in subjaxprs(jaxpr)
    )


_WIDTH_MARKER = re.compile(r"// palladium: threadgroup width (\d+)")


def declared_threadgroup_width(msl_source: str) -> int | None:
    """The thread count a cooperative kernel declares in its
    ``// palladium: threadgroup width N`` line, or None for a kernel that runs one
    thread per program."""
    match = _WIDTH_MARKER.search(msl_source)
    return int(match.group(1)) if match else None


def uses_tensorops(spec: KernelSpec, dot_general: str) -> bool:
    """Whether the dot policy should attempt cooperative recognition.

    True for a kernel containing a dot under "tensorops" or "simdgroup", or under
    "auto" on a 2D or 3D grid. Recognition may still reject the kernel.
    """
    if not _contains(spec.jaxpr, "dot_general"):
        return False
    return dot_general in ("tensorops", "simdgroup") or (
        dot_general == "auto" and len(spec.grid) in (2, 3)
    )
