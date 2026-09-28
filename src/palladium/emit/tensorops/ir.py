"""Structured, layout-aware view of the Pallas jaxpr used by emitter passes."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Var

from palladium.errors import EmitError

from .plan import AddressSpace, Distribution, KernelPlan, ValueLayout

type Atom = Var | Literal


@dataclasses.dataclass(frozen=True)
class IRValue:
    """One jaxpr atom paired with its current logical and physical facts."""

    atom: Atom
    layout: ValueLayout | None


@dataclasses.dataclass(frozen=True)
class IROperation:
    """A primitive equation and recursively imported jaxpr regions."""

    equation: JaxprEqn
    operands: tuple[IRValue, ...]
    results: tuple[IRValue, ...]
    regions: tuple[IRRegion, ...]

    @property
    def name(self) -> str:
        return self.equation.primitive.name


@dataclasses.dataclass(frozen=True)
class IRRegion:
    """A block of equations; nested scans and conditionals remain regions."""

    arguments: tuple[IRValue, ...]
    operations: tuple[IROperation, ...]
    results: tuple[IRValue, ...]


@dataclasses.dataclass(frozen=True)
class KernelIR:
    """Imported Pallas kernel plus its still-unmodified structured jaxpr."""

    plan: KernelPlan
    body: IRRegion


def import_kernel(plan: KernelPlan) -> KernelIR:
    """Import a Pallas kernel into nested regions and typed SSA values.

    The importer preserves source equations and their parameters so target
    passes can inspect primitive-specific metadata without losing JAX
    semantics. Pallas refs at the top level receive the layouts established
    by `KernelPlan`; ordinary intermediate values begin with unassigned
    ownership for the layout pass.
    """

    jaxpr = plan.spec.jaxpr
    if jaxpr is None:
        raise EmitError("tensorops import requires a traced Pallas jaxpr")
    root_layouts = plan.inputs + plan.outputs + plan.scratch
    if len(root_layouts) != len(jaxpr.invars):
        raise EmitError("tensorops plan ref layouts do not match the Pallas jaxpr arguments")
    bindings = {var: layout for var, layout in zip(jaxpr.invars, root_layouts, strict=True)}
    return KernelIR(plan, _import_region(jaxpr, bindings))


def _import_region(jaxpr: Jaxpr, incoming: Mapping[Var, ValueLayout]) -> IRRegion:
    values: dict[Var, ValueLayout | None] = dict(incoming)
    arguments = tuple(IRValue(var, values.get(var, _layout(var))) for var in jaxpr.invars)
    operations: list[IROperation] = []
    for equation in jaxpr.eqns:
        operands = tuple(_value(atom, values) for atom in equation.invars)
        results = tuple(IRValue(atom, _layout(atom)) for atom in equation.outvars)
        values.update(
            (atom, result.layout)
            for atom, result in zip(equation.outvars, results, strict=True)
            if isinstance(atom, Var)
        )
        regions = tuple(_import_region(child, {}) for child in _nested_jaxprs(equation.params))
        operations.append(IROperation(equation, operands, results, regions))
    return IRRegion(
        arguments,
        tuple(operations),
        tuple(_value(atom, values) for atom in jaxpr.outvars),
    )


def _value(atom: Atom, values: Mapping[Var, ValueLayout | None]) -> IRValue:
    if isinstance(atom, Literal):
        return IRValue(atom, _layout(atom))
    return IRValue(atom, values.get(atom, _layout(atom)))


def _layout(atom: Atom) -> ValueLayout | None:
    aval = atom.aval
    if isinstance(aval, ShapedArray):
        return ValueLayout(
            tuple(int(dim) for dim in aval.shape),
            aval.dtype.name,
            AddressSpace.THREAD,
            Distribution.UNASSIGNED,
        )
    inner = getattr(aval, "inner_aval", None)
    if isinstance(inner, ShapedArray):
        return ValueLayout(
            tuple(int(dim) for dim in inner.shape),
            inner.dtype.name,
            AddressSpace.DEVICE,
            Distribution.UNASSIGNED,
        )
    return None


def _nested_jaxprs(value) -> tuple[Jaxpr, ...]:
    found: list[Jaxpr] = []

    def visit(item) -> None:
        if isinstance(item, ClosedJaxpr):
            found.append(item.jaxpr)
        elif isinstance(item, Jaxpr):
            found.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found)
