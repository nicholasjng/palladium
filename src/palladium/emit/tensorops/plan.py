"""Target-independent execution and value-layout plans for the tensorops emitter.

This is the boundary between a traced Pallas kernel and target lowering. It
keeps the Pallas program grid distinct from the Metal execution scope: a
program may execute as one MSL thread or as a cooperating threadgroup.
"""

from __future__ import annotations

import dataclasses
import enum

from palladium.errors import EmitError
from palladium.trace import KernelSpec

from ._shared import SIMDGROUPS


class ProgramScope(enum.Enum):
    """Metal execution unit assigned to one logical Pallas grid point."""

    THREAD = "thread"
    THREADGROUP = "threadgroup"


class AddressSpace(enum.Enum):
    """Storage location of a lowered value."""

    DEVICE = "device"
    THREAD = "thread"
    THREADGROUP = "threadgroup"


class Distribution(enum.Enum):
    """Ownership policy for elements of a value inside its address space."""

    UNASSIGNED = "unassigned"
    REPLICATED = "replicated"
    FLAT_STRIDED = "flat_strided"
    ROW_STRIDED = "row_strided"
    TENSOROPS = "tensorops"


@dataclasses.dataclass(frozen=True)
class ValueLayout:
    """Logical tensor type plus its eventual storage and cooperative owner.

    `UNASSIGNED` is intentional at the frontend boundary. Layout assignment
    passes can choose an ownership policy later without changing the Pallas
    jaxpr or pretending that every value is already a per-thread array.
    """

    shape: tuple[int, ...]
    dtype: str
    address_space: AddressSpace
    distribution: Distribution = Distribution.UNASSIGNED

    def __post_init__(self) -> None:
        if any(dim < 0 for dim in self.shape):
            raise ValueError("value layout dimensions must be nonnegative")


@dataclasses.dataclass(frozen=True)
class KernelPlan:
    """Initial tensorops plan retaining Pallas semantics for target-specific passes."""

    spec: KernelSpec
    scope: ProgramScope
    simdgroups: int
    inputs: tuple[ValueLayout, ...]
    outputs: tuple[ValueLayout, ...]
    scratch: tuple[ValueLayout, ...]

    @property
    def grid(self) -> tuple[int, ...]:
        """Logical Pallas grid, before mapping it to Metal launch geometry."""

        return self.spec.grid

    @property
    def program_id_attribute(self) -> str:
        """MSL builtin used to obtain the logical Pallas program coordinate."""

        if self.scope is ProgramScope.THREADGROUP:
            return "threadgroup_position_in_grid"
        return "thread_position_in_grid"


def plan_kernel(spec: KernelSpec, *, scope: ProgramScope | None = None) -> KernelPlan:
    """Build the tensorops frontend plan without selecting physical value layouts.

    Kernels that explicitly use thread indices, threadgroup size, barriers, or
    threadgroup scratch require group execution. Other kernels retain the
    existing one-thread-per-program default. TensorOps-specific callers may
    request group execution explicitly.
    """

    if len(spec.grid) > 3:
        raise EmitError(f"Metal supports at most a 3D logical grid, got {spec.grid}")
    inferred_scope = ProgramScope.THREADGROUP if spec.uses_threadgroup else ProgramScope.THREAD
    selected_scope = scope or inferred_scope
    if spec.uses_threadgroup and selected_scope is ProgramScope.THREAD:
        raise EmitError("cooperative Pallas effects require one threadgroup per program")
    return KernelPlan(
        spec=spec,
        scope=selected_scope,
        simdgroups=SIMDGROUPS if selected_scope is ProgramScope.THREADGROUP else 1,
        inputs=tuple(_ref_layout(info, AddressSpace.DEVICE) for info in spec.inputs),
        outputs=tuple(_ref_layout(info, AddressSpace.DEVICE) for info in spec.outputs),
        scratch=tuple(
            ValueLayout(tuple(item.shape), item.dtype.name, _scratch_space(item.space))
            for item in spec.scratch
        ),
    )


def _ref_layout(info, space: AddressSpace) -> ValueLayout:
    shape = info.full_block_shape
    if shape is None:
        missing = len(info.array_shape) - len(info.block_shape)
        shape = (1,) * missing + info.block_shape
    return ValueLayout(
        tuple(1 if dim is None else int(dim) for dim in shape), info.dtype.name, space
    )


def _scratch_space(space: str) -> AddressSpace:
    if space == "threadgroup":
        return AddressSpace.THREADGROUP
    return AddressSpace.THREAD
