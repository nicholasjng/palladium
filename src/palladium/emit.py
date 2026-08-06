"""jaxpr -> MSL: palladium's core emitter.

Emission model: scalarized SSA over thread-local arrays. One Metal thread
runs one Pallas program instance; `program_id(k)` maps to a component of
`thread_position_in_grid`. Every jaxpr variable of shape S becomes a
thread-local C array of prod(S) elements (scalar for rank 0); every rule
emits plain element loops. Nothing is vectorized.

Blocks must be contiguous in the underlying array: block shape must equal
array shape on every dim but the leading one.

Per-primitive rules live in RULES, keyed by `eqn.primitive.name`,
registered with @rule(...).

`palladium.debug_msl` prints the emitted MSL; `metal_call(...).interpret`
is the CPU oracle.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import math
import string
from collections.abc import Callable, Iterator

from jax.core import ShapedArray
from jax.extend.core import Jaxpr, JaxprEqn, Literal, Var

from palladium.trace import BlockInfo, KernelSpec

__all__ = ["RULES", "CVal", "EmitError", "EmitState", "emit_jaxpr", "emit_msl", "rule"]

# eqn.invars entries are Vars or inline Literals; eqn.outvars are always Vars.
Atom = Var | Literal


# limit compute primitive arity to a maximum of 6.
MAX_PRIMITIVE_ARITY = 6
PRIMITIVE_INVARS = string.ascii_lowercase[:MAX_PRIMITIVE_ARITY]


def _template_fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def _unwrapped(expr: str) -> str:
    # The RHS of an assignment is syntactically complete, so one wrapping
    # paren pair is always redundant there; stripping it keeps the emitted
    # text readable. Safe only at assign sites: template parens elsewhere
    # defend operand atomicity (see the ELEMENTWISE comment).
    if not (expr.startswith("(") and expr.endswith(")")):
        return expr
    depth = 0
    for i, ch in enumerate(expr):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(expr) - 1:
            # The leading paren closes early: shapes like (a) * (b).
            return expr
    return expr[1:-1]


def _shaped(aval: object) -> ShapedArray:
    # Invariant, not a hope: every non-Ref value in a Pallas kernel jaxpr
    # is shaped, and Refs never pass through declare()/val().
    assert isinstance(aval, ShapedArray), aval
    return aval


CTYPES = {
    "float32": "float",
    "float16": "half",
    "bfloat16": "bfloat",
    "int32": "int",
    "uint32": "uint",
    "bool": "bool",
}

_PID = ("_pid.x", "_pid.y", "_pid.z")


class EmitError(Exception):
    """The emitter cannot lower this kernel (unsupported or invalid input).

    Distinct from NotImplementedError, which marks a missing rule.
    """


@dataclasses.dataclass(frozen=True)
class CVal:
    """A jaxpr atom lowered to C.

    Attributes
    ----------
    expr : str
        C identifier, literal, or expression; any valid rvalue.
    shape : tuple of int
        Logical shape; `()` for scalars.
    ctype : str
        C type name, always a value of CTYPES (`"float"`, not `"float32"`).
    """

    expr: str
    shape: tuple[int, ...]
    ctype: str

    @property
    def size(self) -> int:
        """Element count; 1 for scalars (`math.prod(()) == 1`)."""
        return math.prod(self.shape)

    def at(self, index: str) -> str:
        """Element access that absorbs rank-0/rank-N mixing.

        Scalars ignore the index; arrays apply it. This is all the
        broadcasting the emitter does.

        Parameters
        ----------
        index : str
            C expression for the element index; ignored for scalars.

        Returns
        -------
        str
            `expr` for scalars, `expr[index]` for arrays.
        """
        return self.expr if not self.shape else f"{self.expr}[{index}]"


class EmitState:
    """Emission state: MSL lines and the jaxpr-var -> CVal environment.

    Attributes
    ----------
    lines : list of str
        Emitted MSL body lines, indented.
    env : dict
        Maps jaxpr Vars to their CVals.
    indent : int
        Current indentation depth, managed by `block`.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.env: dict[Var, CVal] = {}
        self._names = itertools.count()
        self.indent = 1

    def emit(self, line: str) -> None:
        """Append one MSL line at the current indentation."""
        self.lines.append("    " * self.indent + line)

    def fresh(self, prefix: str = "t") -> str:
        """Return a new unique C identifier; deterministic per process order."""
        return f"{prefix}{next(self._names)}"

    def val(self, atom: Atom) -> CVal:
        """Resolve a jaxpr atom to its CVal.

        Vars are looked up in the environment; Literals are formatted in
        place.

        Parameters
        ----------
        atom : Var or Literal
            An equation operand.

        Returns
        -------
        CVal
        """
        if isinstance(atom, Literal):
            ctype = CTYPES[str(_shaped(atom.aval).dtype)]
            v = atom.val
            if math.isinf(v):
                expr = "-INFINITY" if v < 0 else "INFINITY"
            elif math.isnan(v):
                expr = "NAN"
            else:
                expr = f"{float(v)!r}f" if ctype in ("float", "half") else str(int(v))
            return CVal(expr=expr, shape=(), ctype=ctype)
        return self.env[atom]

    def declare(self, var: Var) -> CVal:
        """Emit thread-local storage for `var` and bind it.

        Use for a rule producing a new value; use `bind` for aliasing.

        Parameters
        ----------
        var : Var
            A jaxpr output variable; its aval fixes shape and ctype.

        Returns
        -------
        CVal
            The declared storage, already bound in `env`.
        """
        aval = _shaped(var.aval)
        ctype = CTYPES[str(aval.dtype)]
        shape = tuple(int(d) for d in aval.shape)
        name = self.fresh()
        size = math.prod(shape)
        self.emit(f"{ctype} {name}[{size}];" if shape else f"{ctype} {name};")
        cval = CVal(expr=name, shape=shape, ctype=ctype)
        self.env[var] = cval
        return cval

    def bind(self, var: Var, cval: CVal) -> CVal:
        """Bind `var` to an existing CVal: aliasing, no declaration.

        Returns
        -------
        CVal
            `cval`, unchanged, for chaining.
        """
        self.env[var] = cval
        return cval

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        """Emit a braced, indented block: `with state.block("for (...)"):`."""
        self.emit(header + " {")
        self.indent += 1
        yield
        self.indent -= 1
        self.emit("}")

    def copy(self, dst: CVal, src: CVal, count: int) -> None:
        """Emit `count` element assignments dst[i] = src[i] as a loop."""
        i = self.fresh("_i")
        with self.block(f"for (uint {i} = 0; {i} < {count}; ++{i})"):
            self.emit(f"{dst.at(i)} = {src.at(i)};")


RuleFn = Callable[[EmitState, JaxprEqn], None]

RULES: dict[str, RuleFn] = {}


def rule(*names: str) -> Callable[[RuleFn], RuleFn]:
    """Register a lowering rule for one or more primitive names.

    Parameters
    ----------
    *names : str
        Primitive names (`eqn.primitive.name`) the rule handles.

    Returns
    -------
    callable
        Decorator returning the rule unchanged.
    """

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def emit_jaxpr(state: EmitState, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a jaxpr, dispatching each equation to RULES.

    Nested jaxprs (scan bodies, index maps) recurse through this same
    entry point.

    Parameters
    ----------
    state : EmitState
        Emission state; lines are appended in place.
    jaxpr : Jaxpr
        The (sub-)jaxpr to lower.
    in_vals : list of CVal
        Bindings for `jaxpr.invars`, in order.

    Returns
    -------
    list of CVal
        The values of `jaxpr.outvars`.

    Raises
    ------
    EmitError
        If the jaxpr captures arrays as constvars.
    NotImplementedError
        If an equation has no registered rule.
    """
    if jaxpr.constvars:
        raise EmitError(
            "kernel jaxpr has constvars (captured arrays); close over Python "
            "scalars only, or pass arrays as kernel operands"
        )
    for var, cval in zip(jaxpr.invars, in_vals, strict=True):
        state.env[var] = cval
    for eqn in jaxpr.eqns:
        impl = RULES.get(eqn.primitive.name)
        if impl is None:
            raise NotImplementedError(
                f"no MSL rule for primitive '{eqn.primitive.name}'; add one "
                "with @rule(...) in emit.py (planned coverage: ROADMAP)"
            )
        impl(state, eqn)
    return [state.val(v) for v in jaxpr.outvars]


def _check_contiguous(info: BlockInfo, what: str) -> None:
    # Any partial trailing dim makes the block strided in memory, which
    # the flat pointer-plus-offset model cannot express.
    full_block = _full_block_shape(info)
    if any(b != a for b, a in zip(full_block[1:], info.array_shape[1:], strict=True)):
        raise EmitError(
            f"{what}: block {info.block_shape} of array {info.array_shape} is "
            "not contiguous; only the leading block dim may be partial"
        )


def _constant_offset(info: BlockInfo) -> str | None:
    """Fold index maps with no inputs (gridless or constant) to an offset."""
    imj = info.index_map_jaxpr.jaxpr
    if imj.invars or imj.eqns:
        return None
    strides = _element_strides(info.array_shape)
    full_block = _full_block_shape(info)
    off = 0
    for o, b, s in zip(imj.outvars, full_block, strides, strict=True):
        # No invars and no eqns leaves only inline constants as outputs.
        assert isinstance(o, Literal), o
        off += int(o.val) * b * s
    return str(off)


def _element_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[d + 1 :]) for d in range(len(shape)))


def _full_block_shape(info: BlockInfo) -> tuple[int, ...]:
    """block_shape with squeezed dims restored as 1, rank-matched to array."""
    missing = len(info.array_shape) - len(info.block_shape)
    return (1,) * missing + info.block_shape


def emit_msl(spec: KernelSpec, kernel_name: str | None = None) -> str:
    """Assemble the full MSL source for a KernelSpec.

    Signature convention (relied on by `dispatch.bind`): operands in
    jaxpr order (inputs then outputs) bound to `[[buffer(k)]]`, then
    `uint3 _pid [[thread_position_in_grid]]`.

    Parameters
    ----------
    spec : KernelSpec
        Traced kernel, from `palladium.trace`.
    kernel_name : str, optional
        Overrides `spec.name` as the MSL function name.

    Returns
    -------
    str
        Complete, self-contained MSL source.

    Raises
    ------
    EmitError
        For grids over rank 3, non-contiguous blocks, or scratch operands.
    NotImplementedError
        If the kernel stages a primitive with no registered rule.
    """
    name = kernel_name or spec.name
    if len(spec.grid) > 3:
        raise EmitError(f"grid {spec.grid} has rank > 3; Metal grids are 3D")

    operands = list(spec.inputs) + list(spec.outputs)
    n_in = len(spec.inputs)
    params = []
    for k, info in enumerate(operands):
        _check_contiguous(info, f"operand {k}")
        qual = "device" if k >= n_in else "const device"
        ctype = CTYPES[info.dtype.name]
        params.append(f"{qual} {ctype}* arg{k}_base [[buffer({k})]]")
    params.append("uint3 _pid [[thread_position_in_grid]]")

    state = EmitState()
    ref_vals: list[CVal] = []
    for k, info in enumerate(operands):
        qual = "device" if k >= n_in else "const device"
        ctype = CTYPES[info.dtype.name]
        offset = _constant_offset(info)
        if offset is None:
            offset = _block_offset(state, spec, info)
        state.emit(f"{qual} {ctype}* arg{k} = arg{k}_base + {offset};")
        ref_vals.append(CVal(expr=f"arg{k}", shape=info.block_shape, ctype=ctype))

    n_refs = len(operands)
    if len(spec.jaxpr.invars) != n_refs:
        raise EmitError(
            f"kernel has {len(spec.jaxpr.invars)} refs but spec carries "
            f"{n_refs} operands; scratch_shapes are not supported yet"
        )
    emit_jaxpr(state, spec.jaxpr, ref_vals)

    head = f"kernel void {name}(\n    " + ",\n    ".join(params) + ")\n{"
    return "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            head,
            *state.lines,
            "}",
            "",
        ]
    )


@rule("get")
def _rule_get(state: EmitState, eqn: JaxprEqn) -> None:
    """Load a full block from a Ref: `y = x_ref[...]`.

    Indexed loads (`x_ref[i]`) are unsupported.
    """
    if eqn.params["tree"].num_leaves:
        raise NotImplementedError("general pytrees are out of scope for now")

    src = state.val(eqn.invars[0])
    dst = state.declare(eqn.outvars[0])
    state.copy(dst, src, dst.size)


@rule("swap")
def _rule_swap(state: EmitState, eqn: JaxprEqn) -> None:
    """Store a full block to a Ref: `o_ref[...] = y`.

    Deviation: the outvar binds to the ref itself, not a pre-store
    snapshot, so reads of a swap result alias device memory.
    """
    if eqn.params["tree"].num_leaves:
        raise NotImplementedError("general pytrees are out of scope for now")

    ref, value = eqn.invars
    dst_ref = state.val(ref)
    stored = state.val(value)
    state.copy(dst_ref, stored, dst_ref.size)
    state.bind(eqn.outvars[0], dst_ref)


@rule("jit")
def _inline_jit(state: EmitState, eqn: JaxprEqn) -> None:
    """Inline a jit-wrapped call by walking its body jaxpr."""
    inner: Jaxpr = eqn.params["jaxpr"]
    body = inner.jaxpr
    if inner.consts:
        raise EmitError("jit with consts is unsupported")

    invals = [state.val(invar) for invar in eqn.invars]
    outvals = emit_jaxpr(state, body, invals)

    for outvar, val in zip(eqn.outvars, outvals, strict=True):
        state.bind(outvar, val)


# MSL templates for pure elementwise primitives; {a}/{b} are element
# expressions. The arity check in _rule_elementwise validates every template
# against its equation. fmin/fmax are float-only; integer min/max needs a
# ctype dispatch (stretch-7 pre-flight, see ROADMAP).
# The parens are deliberate: CVal.expr admits any rvalue, so a template
# must not assume its operands bind tighter than its own operator. Only
# the assign site may strip them (_unwrapped).
ELEMENTWISE: dict[str, str] = {
    # binary
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "min": "fmin({a}, {b})",
    "max": "fmax({a}, {b})",
    "pow": "pow({a}, {b})",
    # unary
    "neg": "-{a}",
    "abs": "fabs({a})",
    "exp": "exp({a})",
    "log": "log({a})",
    "sin": "sin({a})",
    "cos": "cos({a})",
    "sqrt": "sqrt({a})",
    "tanh": "tanh({a})",
    # ternary
    "select_n": "({a} ? {c} : {b})",  # a: predicate, c when true, b when false
    # logical
    "lt": "({a} < {b})",
    "le": "({a} <= {b})",
    "gt": "({b} < {a})",
    "ge": "({b} <= {a})",
}


def _rule_elementwise(state: EmitState, eqn: JaxprEqn) -> None:
    """One rule for every pure elementwise primitive.

    Three templates are derived rather than looked up: integer_pow expands
    to repeated multiplication (reciprocal for negative y); convert_element_type
    casts to the output's ctype; broadcast_in_dim is "{a}" (the element
    loop already broadcasts scalars).

    The arity check requires every template field to be filled and every
    operand consumed.
    """

    dst = state.declare(eqn.outvars[0])
    ops = [state.val(v) for v in eqn.invars]
    opname = eqn.primitive.name

    if opname == "integer_pow":
        exp: int = eqn.params["y"]
        body = " * ".join(["{a}"] * abs(exp))
        template = f"({body})" if exp > 0 else f"(1.0f / ({body}))"
    elif opname == "convert_element_type":
        template = f"(({dst.ctype}){{a}})"
    elif opname == "broadcast_in_dim":
        if dst.shape and any(op.shape for op in ops):
            raise EmitError("array-to-array broadcasts are unimplemented")
        template = "{a}"
    elif opname == "select_n":
        if (pred_type := ops[0].ctype) != "bool":
            raise EmitError(
                f"select_n requires predicate of type bool, got {pred_type}"
            )
        template = ELEMENTWISE[opname]
    else:
        template = ELEMENTWISE[opname]

    fields = _template_fields(template)
    expected = set(PRIMITIVE_INVARS[: len(ops)])
    if fields != expected:
        raise EmitError(f"{opname} requires {len(fields)} operands, got {len(ops)}")

    def assign(idx: str) -> str:
        inputs = {name: op.at(idx) for name, op in zip(PRIMITIVE_INVARS, ops)}
        return f"{dst.at(idx)} = {_unwrapped(template.format(**inputs))};"

    if dst.shape:
        idx = state.fresh("_i")
        with state.block(f"for (uint {idx} = 0; {idx} < {dst.size}; ++{idx})"):
            state.emit(assign(idx))
    else:
        state.emit(assign("0"))


for _name in [*ELEMENTWISE, "integer_pow", "convert_element_type", "broadcast_in_dim"]:
    RULES[_name] = _rule_elementwise


@rule("program_id")
def _rule_program_id(state: EmitState, eqn: JaxprEqn) -> None:
    """`pl.program_id(axis)` -> a component of _pid, as a rank-0 int.

    Pure aliasing, no storage or code. The (int) cast keeps index
    arithmetic signed.
    """
    axis: int = eqn.params["axis"]
    state.bind(eqn.outvars[0], CVal(f"(int){_PID[axis]}", (), "int"))


def _block_offset(state: EmitState, spec: KernelSpec, info: BlockInfo) -> str:
    """Element offset of this program instance's block, as a C expression.

    The index map is a jaxpr over grid indices (bound to _pid components),
    recursed through emit_jaxpr. Map outputs are block indices per array
    dim, converted to elements as

        offset = sum(idx[d] * full_block[d] * stride[d] for d in dims)

    Zero-literal terms are elided; falls back to "0" for an all-zero map.
    """
    pid_vals = [CVal(f"(int){_PID[k]}", (), "int") for k in range(len(spec.grid))]
    out_vals = emit_jaxpr(state, info.index_map_jaxpr.jaxpr, pid_vals)
    block_shape = _full_block_shape(info)
    strides = _element_strides(info.array_shape)

    offsets = []
    for val, shape, stride in zip(out_vals, block_shape, strides, strict=True):
        if val.expr == "0":
            continue
        offsets.append(f"{val.expr} * {shape * stride}")

    return " + ".join(offsets) or "0"


@rule("scan")
def _rule_scan(state: EmitState, eqn: JaxprEqn) -> None:
    """`lax.fori_loop` / pure-carry `lax.scan` -> a C for-loop.

    fori_loop stages as scan with no xs/ys, usually with consts ordered
    before carries in both eqn.invars and body.invars. Consts pass through
    unchanged; carries get fresh mutable loop variables bound on the
    outvars.

    Copy-back runs in two phases: scan updates all carries simultaneously,
    so phase 1 snapshots reads that alias other carries into temps
    (skipping self-forward no-ops), then phase 2 overwrites the carries.
    """
    num_carry = len(eqn.outvars)
    num_consts = len(eqn.invars) - num_carry
    length: int = eqn.params["length"]
    body: Jaxpr = eqn.params["jaxpr"]

    if len(body.outvars) != num_carry:
        raise EmitError("lax.scan with stacked ys not supported, use lax.fori_loop")

    reverse: bool = eqn.params.get("reverse", False)
    if reverse:
        raise EmitError("no reverse-mode scan support, use lax.fori_loop")

    const_vals = [state.val(v) for v in eqn.invars[:num_consts]]

    carries = []
    for invar, outvar in zip(eqn.invars[num_consts:], eqn.outvars, strict=True):
        dst = state.declare(outvar)
        state.copy(dst, state.val(invar), dst.size)
        carries.append(dst)

    carry_exprs = {c.expr for c in carries}
    idx = state.fresh("_s")
    with state.block(f"for (uint {idx} = 0; {idx} < {length}; ++{idx})"):
        outs = emit_jaxpr(state, body, const_vals + carries)
        sources = []
        for out, carry in zip(outs, carries, strict=True):
            if out.expr == carry.expr:
                # Self-forward: the value is already in place.
                sources.append(None)
            elif out.expr in carry_exprs:
                # Aliases a different carry: snapshot before any overwrite.
                tmp = CVal(state.fresh(), out.shape, out.ctype)
                state.emit(
                    f"{tmp.ctype} {tmp.expr}[{tmp.size}];"
                    if tmp.shape
                    else f"{tmp.ctype} {tmp.expr};"
                )
                state.copy(tmp, out, out.size)
                sources.append(tmp)
            else:
                # Fresh SSA name: no carry write can clobber it.
                sources.append(out)

        for src, carry in zip(sources, carries, strict=True):
            if src is not None:
                state.copy(carry, src, carry.size)
