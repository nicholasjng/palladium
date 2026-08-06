"""jaxpr -> MSL: palladium's core emitter.

Emission model: scalarized SSA over thread-local arrays. One Metal thread
runs one Pallas program instance; `program_id(k)` maps to a component of
`thread_position_in_grid`. Every jaxpr variable of shape S becomes a
thread-local C array of prod(S) elements (a bare scalar for rank 0), and
every rule emits plain element loops. Nothing is vectorized: the Metal
compiler cleans up after us, and for ensemble workloads (a block is one
small ODE state) one-thread-per-program is the right mapping. Triton maps
a program to a threadgroup and Mosaic GPU to a warpgroup; this tier is
deliberately the bottom rung of that scope ladder, and survives unchanged
if the plmetal dialect (ROADMAP, backend endgame) ever lands.

Blocks must be contiguous in the underlying array: the block shape must
equal the array shape on every dim but the leading one. Lifting this means
strided copy loops in get/swap (ROADMAP stretch 6).

Per-primitive rules live in RULES, keyed by `eqn.primitive.name` and
registered with @rule(...). The file grew rule-by-rule through the
test-driven exercise arc in ROADMAP.md; tests/test_02..06 pin each rule,
and known gaps are tracked there (indexed ref access, select_n, inf/nan
literals, integer min/max, integer_pow y=0).

Debugging: the output is text, read it. `palladium.debug_msl` is the
one-call version; `metal_call(...).interpret` is the CPU oracle.
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

__all__ = ["RULES", "CVal", "Ctx", "EmitError", "emit_jaxpr", "emit_msl", "rule"]

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

    Distinct from NotImplementedError, which marks a missing rule: an
    EmitError means the construct is recognized and rejected.
    """


@dataclasses.dataclass(frozen=True)
class CVal:
    """A jaxpr atom lowered to C.

    Attributes
    ----------
    expr : str
        C identifier, literal, or expression. Any valid rvalue works;
        `program_id` binds a bare `(int)_pid.x`, for example.
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

        Scalars ignore the index and arrays apply it, so rules can emit
        `dst.at(i) = a.at(i) * b.at(i)` without caring which operands are
        scalars. This is all the broadcasting the emitter does.

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


class Ctx:
    """Emission state: MSL lines and the jaxpr-var -> CVal environment.

    Attributes
    ----------
    lines : list of str
        Emitted MSL body lines, indented.
    env : dict
        Maps jaxpr Vars to their CVals. SSA order guarantees lookups
        succeed: every Var was an earlier equation's output or an input.
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
        """Return a new unique C identifier.

        The counter is process-order deterministic, which keeps golden-MSL
        snapshots stable.
        """
        return f"{prefix}{next(self._names)}"

    def val(self, atom: Atom) -> CVal:
        """Resolve a jaxpr atom to its CVal.

        The only bridge from jaxpr land to C land: Vars are looked up in
        the environment, Literals are formatted in place.

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

        Use when a rule produces a new value; use `bind` when it aliases
        an existing one.

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

        No duplicate-binding assert on purpose: a body jaxpr shared by two
        equations legitimately rebinds its invars, and SSA order makes
        rebinding harmless.

        Returns
        -------
        CVal
            `cval`, unchanged, for chaining.
        """
        self.env[var] = cval
        return cval

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        """Emit a braced, indented block: `with ctx.block("for (...)"):`."""
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


RuleFn = Callable[[Ctx, JaxprEqn], None]

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
        Decorator returning the rule unchanged. The registry is a plain
        dict; assigning `RULES[name] = fn` directly is equally valid.
    """

    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def emit_jaxpr(ctx: Ctx, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a jaxpr, dispatching each equation to RULES.

    Rules for nested jaxprs (scan bodies, index maps) recurse through this
    same entry point, so anything the rules support composes.

    Parameters
    ----------
    ctx : Ctx
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
        ctx.env[var] = cval
    for eqn in jaxpr.eqns:
        impl = RULES.get(eqn.primitive.name)
        if impl is None:
            raise NotImplementedError(
                f"no MSL rule for primitive '{eqn.primitive.name}'; add one "
                "with @rule(...) in emit.py (planned coverage: ROADMAP)"
            )
        impl(ctx, eqn)
    return [ctx.val(v) for v in jaxpr.outvars]


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

    Signature convention, relied on by `dispatch.bind`: kernel operands in
    jaxpr order (inputs then outputs) bound to `[[buffer(k)]]` in that same
    order, then `uint3 _pid [[thread_position_in_grid]]`.

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

    ctx = Ctx()
    ref_vals: list[CVal] = []
    for k, info in enumerate(operands):
        qual = "device" if k >= n_in else "const device"
        ctype = CTYPES[info.dtype.name]
        offset = _constant_offset(info)
        if offset is None:
            offset = _block_offset(ctx, spec, info)
        ctx.emit(f"{qual} {ctype}* arg{k} = arg{k}_base + {offset};")
        ref_vals.append(CVal(expr=f"arg{k}", shape=info.block_shape, ctype=ctype))

    n_refs = len(operands)
    if len(spec.jaxpr.invars) != n_refs:
        raise EmitError(
            f"kernel has {len(spec.jaxpr.invars)} refs but spec carries "
            f"{n_refs} operands; scratch_shapes are not supported yet"
        )
    emit_jaxpr(ctx, spec.jaxpr, ref_vals)

    head = f"kernel void {name}(\n    " + ",\n    ".join(params) + ")\n{"
    return "\n".join(
        [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            head,
            *ctx.lines,
            "}",
            "",
        ]
    )


@rule("get")
def _rule_get(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Load a full block from a Ref: `y = x_ref[...]`.

    Indexed loads like `x_ref[i]` (a non-empty indexer tree) are ROADMAP
    stretch 6.
    """
    if eqn.params["tree"].num_leaves:
        raise NotImplementedError("general pytrees are out of scope for now")

    src = ctx.val(eqn.invars[0])
    dst = ctx.declare(eqn.outvars[0])
    ctx.copy(dst, src, dst.size)


@rule("swap")
def _rule_swap(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Store a full block to a Ref: `o_ref[...] = y`.

    Known deviation: the outvar is bound to the ref itself, not a snapshot
    of its pre-store contents, so reads of a swap result see the stored
    value and stay aliased to device memory. No kernel reads a swap result
    today; take a temp snapshot before the copy if one ever does.
    """
    if eqn.params["tree"].num_leaves:
        raise NotImplementedError("general pytrees are out of scope for now")

    ref, value = eqn.invars
    dst_ref = ctx.val(ref)
    stored = ctx.val(value)
    ctx.copy(dst_ref, stored, dst_ref.size)
    ctx.bind(eqn.outvars[0], dst_ref)


@rule("jit")
def _inline_jit(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Inline a jit-wrapped call by walking its body jaxpr.

    Inlines a jit body into an MSL program. This is used often, because
    many library functions in jax.numpy are jit-decorated.
    """
    inner: Jaxpr = eqn.params["jaxpr"]
    body = inner.jaxpr
    if inner.consts:
        raise EmitError("jit with consts is unsupported")

    invals = [ctx.val(invar) for invar in eqn.invars]
    outvals = emit_jaxpr(ctx, body, invals)

    for outvar, val in zip(eqn.outvars, outvals, strict=True):
        ctx.bind(outvar, val)


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


def _rule_elementwise(ctx: Ctx, eqn: JaxprEqn) -> None:
    """One rule for every pure elementwise primitive.

    Three templates are derived rather than looked up: integer_pow expands
    to repeated multiplication (a reciprocal for negative y) because the
    exponent lives in params, not operands; convert_element_type casts to
    the output's ctype; broadcast_in_dim is "{a}" because the element loop
    already broadcasts scalars.

    The arity check guards both directions: a template field no operand
    fills, and an operand no field consumes (the silent-drop hazard). The
    "ab" alphabet caps arity at 2 until select_n lands (ROADMAP stretch 7).
    """

    dst = ctx.declare(eqn.outvars[0])
    ops = [ctx.val(v) for v in eqn.invars]
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
        idx = ctx.fresh("_i")
        with ctx.block(f"for (uint {idx} = 0; {idx} < {dst.size}; ++{idx})"):
            ctx.emit(assign(idx))
    else:
        ctx.emit(assign("0"))


for _name in [*ELEMENTWISE, "integer_pow", "convert_element_type", "broadcast_in_dim"]:
    RULES[_name] = _rule_elementwise


@rule("program_id")
def _rule_program_id(ctx: Ctx, eqn: JaxprEqn) -> None:
    """`pl.program_id(axis)` -> a component of _pid, as a rank-0 int.

    Pure aliasing, no storage or code. The (int) cast keeps index
    arithmetic signed: uint underflow in an index map is an out-of-bounds
    read with no diagnostic.
    """
    axis: int = eqn.params["axis"]
    ctx.bind(eqn.outvars[0], CVal(f"(int){_PID[axis]}", (), "int"))


def _block_offset(ctx: Ctx, spec: KernelSpec, info: BlockInfo) -> str:
    """Element offset of this program instance's block, as a C expression.

    The index map is itself a jaxpr taking grid indices in grid-axis
    order; binding them to _pid components and recursing through
    emit_jaxpr lets maps with arithmetic (`i * 16 + j`) reuse the
    elementwise rules. Map outputs are block indices per array dim,
    converted to elements as

        offset = sum(idx[d] * full_block[d] * stride[d] for d in dims)

    Zero-literal terms are elided for legibility of the emitted pointer
    lines; the `or "0"` fallback keeps an all-zero map valid MSL.
    """
    pid_vals = [CVal(f"(int){_PID[k]}", (), "int") for k in range(len(spec.grid))]
    out_vals = emit_jaxpr(ctx, info.index_map_jaxpr.jaxpr, pid_vals)
    block_shape = _full_block_shape(info)
    strides = _element_strides(info.array_shape)

    offsets = []
    for val, shape, stride in zip(out_vals, block_shape, strides, strict=True):
        if val.expr == "0":
            continue
        offsets.append(f"{val.expr} * {shape * stride}")

    return " + ".join(offsets) or "0"


@rule("scan")
def _rule_scan(ctx: Ctx, eqn: JaxprEqn) -> None:
    """`lax.fori_loop` / pure-carry `lax.scan` -> a C for-loop.

    fori_loop stages as scan with no xs/ys but usually with consts: values
    read before the loop ride along as constant operands, ordered before
    the carries in eqn.invars and body.invars alike. Consts pass through
    unchanged; carries get fresh mutable loop variables, declared on the
    outvars so the final values are bound for downstream equations.

    The copy-back runs in two phases because scan updates all carries
    simultaneously while C statements are sequential: a body that forwards
    or permutes its carries stages outputs that alias the carry variables,
    and writing them in order clobbers unread values (tests/test_05:
    test_carry_permutation). Phase 1 snapshots endangered reads into temps
    and skips self-forward no-ops; phase 2 overwrites the carries safely.
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

    const_vals = [ctx.val(v) for v in eqn.invars[:num_consts]]

    carries = []
    for invar, outvar in zip(eqn.invars[num_consts:], eqn.outvars, strict=True):
        dst = ctx.declare(outvar)
        ctx.copy(dst, ctx.val(invar), dst.size)
        carries.append(dst)

    carry_exprs = {c.expr for c in carries}
    idx = ctx.fresh("_s")
    with ctx.block(f"for (uint {idx} = 0; {idx} < {length}; ++{idx})"):
        outs = emit_jaxpr(ctx, body, const_vals + carries)
        sources = []
        for out, carry in zip(outs, carries, strict=True):
            if out.expr == carry.expr:
                # Self-forward: the value is already in place.
                sources.append(None)
            elif out.expr in carry_exprs:
                # Aliases a different carry: snapshot before any overwrite.
                tmp = CVal(ctx.fresh(), out.shape, out.ctype)
                ctx.emit(
                    f"{tmp.ctype} {tmp.expr}[{tmp.size}];"
                    if tmp.shape
                    else f"{tmp.ctype} {tmp.expr};"
                )
                ctx.copy(tmp, out, out.size)
                sources.append(tmp)
            else:
                # Fresh SSA name: no carry write can clobber it.
                sources.append(out)

        for src, carry in zip(sources, carries, strict=True):
            if src is not None:
                ctx.copy(carry, src, carry.size)
