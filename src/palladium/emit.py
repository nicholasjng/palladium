"""jaxpr -> MSL: palladium's core, laid out as test-driven exercises.

## Emission model: scalarized SSA over thread-local arrays

- One Metal thread per Pallas program instance. `dispatch.bind` launches
  `prod(grid)` threads as an up-to-3D grid; `program_id(k)` maps to a
  component of `thread_position_in_grid`. (Triton maps a program to a whole
  threadgroup, Mosaic GPU to a warpgroup; one-thread-per-program is the
  simplest mapping that preserves Pallas semantics, and it is the right one
  for ensemble workloads where a block is one small ODE state. Revisit for
  stencils.) This is also, deliberately, the bottom rung of a scope ladder:
  if the plmetal dialect (ROADMAP, backend endgame) ever lands, a Pallas
  thread becomes a simdgroup or an `execution_simdgroups<N>` scope, and the
  rules in this file become its thread-scope tier unchanged.
- Every jaxpr variable of shape S becomes a thread-local C array of prod(S)
  elements (a bare scalar for rank 0), named t0, t1, ...
- Every primitive rule emits a plain element loop. Nothing is vectorized;
  the Metal compiler cleans up after us.
- Blocks must be *contiguous* in the underlying array: the block shape must
  equal the array shape on every dim but the leading one. `emit_msl` checks
  this. (Lifting it means strided copy loops in get/swap — a stretch
  exercise, see ROADMAP.)

## The exercises

The driver (`emit_msl`, `emit_jaxpr`) and the plumbing (`Ctx`, `CVal`, the
rule registry) are provided and covered by tests/test_01. The per-primitive
rules are yours, in test order — each stub's docstring has the hints:

    exercise 1  _rule_get / _rule_swap           tests/test_02_emit_copy.py
    exercise 2  ELEMENTWISE + _rule_elementwise  tests/test_03_emit_elementwise.py
    exercise 3  _rule_program_id / _block_offset tests/test_04_grid_blocks.py
    exercise 4  _rule_scan                       tests/test_05_loops.py
    exercise 5  composition only, no new rules   tests/test_06_capstone_rk4.py

Run one exercise:  uv run pytest tests/test_02_emit_copy.py -x
Check your MSL:    print(emit_msl(spec)) — it is just text; read it.
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

__all__ = ["CVal", "Ctx", "EmitError", "emit_jaxpr", "emit_msl", "rule"]

# A jaxpr operand: eqn.invars entries are Vars (named intermediates) or
# Literals (inline constants); eqn.outvars are always Vars.
Atom = Var | Literal



def _template_fields(template: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def _shaped(aval: object) -> ShapedArray:
    """Narrow an abstract value to ShapedArray (shape + dtype).

    Every non-Ref value in a Pallas kernel jaxpr is shaped; Refs never pass
    through declare()/val(), so this assert states an invariant, not a hope.
    """
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
    pass


@dataclasses.dataclass(frozen=True)
class CVal:
    """A jaxpr atom lowered to C: an identifier (or literal) plus its type."""

    expr: str
    shape: tuple[int, ...]
    ctype: str

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def at(self, index: str) -> str:
        """Element access that absorbs rank-0/rank-N mixing.

        Scalars ignore the index, arrays apply it — so a rule can emit
        `dst.at(i) = a.at(i) * b.at(i)` without caring which operands are
        scalars. This is all the broadcasting the emitter does.
        """
        return self.expr if not self.shape else f"{self.expr}[{index}]"


class Ctx:
    """Accumulates MSL lines and the jaxpr-var -> CVal environment."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.env: dict[Var, CVal] = {}
        self._names = itertools.count()
        self.indent = 1

    def emit(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def fresh(self, prefix: str = "t") -> str:
        return f"{prefix}{next(self._names)}"

    def val(self, atom: Atom) -> CVal:
        """Look up a Var, or format a Literal in place."""
        if isinstance(atom, Literal):
            ctype = CTYPES[str(_shaped(atom.aval).dtype)]
            v = atom.val
            expr = f"{float(v)!r}f" if ctype in ("float", "half") else str(int(v))
            return CVal(expr=expr, shape=(), ctype=ctype)
        return self.env[atom]

    def declare(self, var: Var) -> CVal:
        """Emit a thread-local declaration for a jaxpr output var."""
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
          """Bind a jaxpr var to an existing CVal, aliasing, no declaration."""
          self.env[var] = cval
          return cval

    @contextlib.contextmanager
    def block(self, header: str) -> Iterator[None]:
        """`with ctx.block("for (...)"):` emits a braced, indented block."""
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


# --- rule registry -----------------------------------------------------------

RuleFn = Callable[[Ctx, JaxprEqn], None]

RULES: dict[str, RuleFn] = {}


def rule(*names: str) -> Callable[[RuleFn], RuleFn]:
    def register(fn: RuleFn) -> RuleFn:
        for n in names:
            RULES[n] = fn
        return fn

    return register


def emit_jaxpr(ctx: Ctx, jaxpr: Jaxpr, in_vals: list[CVal]) -> list[CVal]:
    """Walk a (sub-)jaxpr, dispatching each equation to RULES.

    Binds `jaxpr.invars` to `in_vals`, returns the CVals of `jaxpr.outvars`.
    Rules for nested jaxprs (scan) call this recursively.
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
                f"no MSL rule for primitive '{eqn.primitive.name}' — if this "
                "is one of the exercise stubs, its test names it; otherwise "
                "add a rule with @rule(...) in emit.py"
            )
        impl(ctx, eqn)
    return [ctx.val(v) for v in jaxpr.outvars]


# --- driver (provided) -------------------------------------------------------


def _check_contiguous(info: BlockInfo, what: str) -> None:
    # Only the leading dim may be a partial slice of the array; every
    # trailing dim must span the array fully, or the block is strided.
    full_block = _full_block_shape(info)
    if any(b != a for b, a in zip(full_block[1:], info.array_shape[1:], strict=True)):
        raise EmitError(
            f"{what}: block {info.block_shape} of array {info.array_shape} is "
            "not contiguous; only the leading block dim may be partial"
        )


def _constant_offset(info: BlockInfo) -> str | None:
    """Offset for index maps with no inputs (gridless / constant maps)."""
    imj = info.index_map_jaxpr.jaxpr
    if imj.invars or imj.eqns:
        return None
    strides = _element_strides(info.array_shape)
    full_block = _full_block_shape(info)
    off = 0
    for o, b, s in zip(imj.outvars, full_block, strides, strict=True):
        # No invars and no eqns means every output is an inline constant.
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
    """Assemble the full MSL source for a KernelSpec. Provided.

    Signature convention (dispatch.bind relies on it): kernel operands in
    jaxpr order — inputs then outputs — bound to [[buffer(k)]] in that same
    order, then `uint3 _pid [[thread_position_in_grid]]`.
    """
    name = kernel_name or spec.name
    if len(spec.grid) > 3:
        raise EmitError(f"grid {spec.grid} has rank > 3; Metal grids are 3D")

    operands = list(spec.inputs) + list(spec.outputs)
    n_in = len(spec.inputs)
    params = []
    for k, info in enumerate(operands):
        _check_contiguous(info, f"operand {k}")
        qual = "device" if k >= n_in else "device const"
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
            offset = _block_offset(ctx, spec, info)  # exercise 3
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


# =============================================================================
# Exercises. Everything below is yours. Delete each `raise` and make its test
# green; do not change the provided code above (the tests pin its contract).
# =============================================================================


@rule("get")
def _rule_get(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Exercise 1a — load a block from a Ref: `y = x_ref[...]`.

    Equation shape (see tests/test_01_trace.py output for a live one):
      eqn.invars  = [ref]          ref CVal is a *device pointer* (see driver)
      eqn.outvars = [y]            a thread-local value to declare
      eqn.params["tree"]           the indexer pytree; empty for `[...]`

    Steps:
      * `if eqn.params["tree"].num_leaves: raise NotImplementedError(...)` —
        indexed loads like `x_ref[i]` are the stretch exercise, not this one.
      * `src = ctx.val(eqn.invars[0])`, `dst = ctx.declare(eqn.outvars[0])`
      * copy `dst.size` elements from src to dst (ctx.copy does the loop).
    """
    if eqn.params["tree"].num_leaves:
        raise NotImplementedError("general pytrees are out of scope for now")

    src = ctx.val(eqn.invars[0])
    dst = ctx.declare(eqn.outvars[0])
    ctx.copy(dst, src, dst.size)


@rule("swap")
def _rule_swap(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Exercise 1b — store a block to a Ref: `o_ref[...] = y`.

    Equation shape:
      eqn.invars  = [ref, value]
      eqn.outvars = [old_value]    the swapped-out value; kernels that just
                                   store never read it, but a rule must still
                                   bind every outvar — bind it to the *loaded*
                                   old value only if someone asks (i.e. start
                                   by binding it to the stored value and
                                   revisit if a test ever reads it).
      eqn.params["tree"]           empty for `[...]`, same restriction as get.
    """
    if eqn.params["tree"].num_leaves:
        raise NotImplementedError("general pytrees are out of scope for now")

    ref, val = eqn.invars
    src = ctx.val(ref)
    value = ctx.val(val)
    ctx.copy(src, value, src.size)
    ctx.bind(eqn.outvars[0], src)


# Exercise 2 — the elementwise table. Fill in MSL expression templates keyed
# by primitive name; `{a}`, `{b}` are element expressions. The test tells you
# which primitives it stages; add entries as tests (and later, your own
# kernels) demand them. MSL reference: <metal_stdlib> provides exp, sin, cos,
# sqrt, tanh, log, fabs, fmin/fmax, pow, fma.
ELEMENTWISE: dict[str, str] = {
    "add": "({a} + {b})",
    "mul": "({a} * {b})",
    "div": "({a} / {b})",
    "sub": "({a} - {b})",
    "neg": "-{a}",
    "exp": "exp({a})",
    "sin": "sin({a})",
    "cos": "cos({a})",
    "sqrt": "sqrt({a})",
    "tanh": "tanh({a})",
    "log": "log({a})",
    "abs": "fabs({a})",
    "min": "fmin({a}, {b})",
    "max": "fmax({a}, {b})",
    "pow": "pow({a}, {b})",
}


def _rule_elementwise(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Exercise 2 — one rule for every pure elementwise primitive.

    Steps:
      * template = ELEMENTWISE[eqn.primitive.name]
      * dst = ctx.declare(eqn.outvars[0]); operands via ctx.val(...)
      * one loop over dst.size (rank-0 output: "loop" of one, no brackets —
        CVal.at handles both if you use a real loop only when dst.shape).
      * `dst.at(i) = template.format(a=..., b=...)` — CVal.at makes scalar
        operands (literals like 2.0f, program ids) just work.

    Three primitives need a twist:
      * integer_pow: exponent is in eqn.params["y"]; emit repeated
        multiplication (y is tiny in practice) rather than pow().
      * convert_element_type: cast, `(({ctype}){a})` with the *output* ctype.
      * broadcast_in_dim (scalar -> array, e.g. `jnp.zeros((4,)) + pid`
        stages one): template "{a}" is already correct — CVal.at ignores the
        index on the scalar side. Reject array -> array broadcasts.
    Register the rule once you have it:
        for _name in [*ELEMENTWISE, "integer_pow", "convert_element_type",
                      "broadcast_in_dim"]:
            RULES[_name] = _rule_elementwise
    (a line you add below the function, or extend `rule(...)` usage — your
    call; the registry is just a dict.)
    """

    dst = ctx.declare(eqn.outvars[0])
    ops = [ctx.val(v) for v in eqn.invars]
    opname = eqn.primitive.name

    if opname == "integer_pow":
        exp: int = eqn.params["y"]
        body = " * ".join(["{a}"] * abs(exp))          # "{a} * {a} * {a}"
        template = f"({body})" if exp > 0 else f"(1.0f / ({body}))"
    elif opname == "convert_element_type":
        template = f"(({dst.ctype}){{a}})"
    elif opname == "broadcast_in_dim":
        if dst.shape and any(op.shape for op in ops):
            raise EmitError("array-to-array broadcasts are unimplemented")
        template = "{a}"
    else:
        template = ELEMENTWISE[eqn.primitive.name]

    fields = _template_fields(template)                 # abs -> {"a"}, pow -> {"a", "b"}
    expected = set("ab"[: len(ops)])
    if fields != expected:
        raise EmitError(
            f"{opname}: template references {sorted(fields)}, "
            f"equation has {len(ops)} operand(s)"
        )

    def assign(idx: str) -> str:
          inputs = {name: op.at(idx) for name, op in zip("ab", ops)}
          return f"{dst.at(idx)} = {template.format(**inputs)};"

    if dst.shape:
        idx = ctx.fresh("_i")
        with ctx.block(f"for (uint {idx} = 0; {idx} < {dst.size}; ++{idx})"):
            ctx.emit(assign(idx))
    else:
        ctx.emit(assign("0"))




for _name in [
    *ELEMENTWISE,
    "integer_pow",
    "convert_element_type",
    "broadcast_in_dim"
]:
    RULES[_name] = _rule_elementwise

@rule("program_id")
def _rule_program_id(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Exercise 3a — `pl.program_id(axis)` -> a component of _pid.

    eqn.params["axis"] picks from _PID; bind the outvar to a rank-0 int CVal
    (no declaration needed — a CVal's expr can be any C expression).
    """
    axis: int = eqn.params["axis"]
    ctx.bind(eqn.outvars[0], CVal(f"(int){_PID[axis]}", (), "int"))


def _block_offset(ctx: Ctx, spec: KernelSpec, info: BlockInfo) -> str:
    """Exercise 3b — element offset of this program's block, as a C expr.

    Called by the driver for every operand whose index map actually uses the
    grid indices (constant maps are already handled).

    The index map is itself a tiny jaxpr: `lambda i: (i, 0)` has
    invars=[i], no eqns, outvars=[i, Literal 0]. Its inputs are the grid
    indices in grid-axis order — bind invar k to CVal(_PID[k], (), "int")
    and evaluate it with emit_jaxpr (maps with arithmetic in them then work
    for free, using your exercise-2 rules). The result is the *block index*
    per array dim; convert to elements with

        offset = sum(idx[d] * full_block[d] * stride[d] for d in dims)

    using _full_block_shape(info) and _element_strides(info.array_shape) —
    build the sum as a C expression string, constants folded or not, the
    Metal compiler does not care.
    """
    pid_vals = [CVal(f"(int){_PID[k]}", (), "int") for k in range(len(spec.grid))]

    # block indices per array dimension
    out_vals = emit_jaxpr(ctx, info.index_map_jaxpr.jaxpr, pid_vals)
    block_shape = _full_block_shape(info)
    strides = _element_strides(info.array_shape)

    offsets = []
    for (val, shape, stride) in zip(out_vals, block_shape, strides, strict=True):
        # elide 0 literals from output index calculations.
        # shape * stride != 0 except for zero-sized arrays,
        # which are uninteresting.
        if val.expr == "0":
            continue
        offsets.append(f"{val.expr} * {shape * stride}")

    return " + ".join(offsets) or "0"




@rule("scan")
def _rule_scan(ctx: Ctx, eqn: JaxprEqn) -> None:
    """Exercise 4 — `lax.fori_loop` / pure-carry `lax.scan` -> a C for-loop.

    fori_loop stages as scan with no xs/ys, but usually WITH consts: any
    value read before the loop and used inside it (per-thread ODE
    parameters, say) rides along as a constant operand. Operand order is
    scan's invariant: consts first, then carries.

      eqn.outvars            final carries -> num_carry = len(eqn.outvars)
      eqn.invars             [*consts, *carries]; num_consts = the rest
      eqn.params["length"]   trip count (a Python int — loops are static)
      eqn.params["jaxpr"]    the body as a ClosedJaxpr; body = param.jaxpr
                             body.invars = [*consts, *carries] too
      eqn.params["reverse"]  raise EmitError if True (fori never sets it)

    A scan with ys (outvars beyond the carries — i.e. a real lax.scan that
    stacks per-step outputs) is out of scope: detect it by body.outvars
    being longer than eqn.outvars... it cannot happen here, because ys
    would make len(body.outvars) > num_carry; guard with an EmitError if
    you want the sharp edge labeled.

    Steps:
      * consts: bind directly — `const_vals = [ctx.val(v) for v in
        eqn.invars[:num_consts]]`; they are never written, no copies needed.
      * carries: declare a mutable loop var per carry (ctx.declare on each
        *outvar* gives correctly-typed storage), copy the initial values in
        (ctx.copy from the trailing invars).
      * `with ctx.block(f"for (uint _s = 0; _s < {length}; ++_s)"):`
          - run the body: `outs = emit_jaxpr(ctx, body, const_vals +
            carry_vals)` — fresh SSA names inside the loop are re-declared
            every iteration; MSL is fine with that;
          - copy `outs` back into the carry vars (the loop-carried
            dependence; do NOT skip it — the body's outputs are new SSA
            names, not your loop vars).
      * the carry vars are already bound to eqn.outvars if you declared
        them via ctx.declare(outvar). Done.
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

    # bind consts directly
    const_vals = [ctx.val(v) for v in eqn.invars[:num_consts]]

    carries = []
    for invar, outvar in zip(eqn.invars[num_consts:], eqn.outvars, strict=True):
        dst = ctx.declare(outvar)
        ctx.copy(dst, ctx.val(invar), dst.size)
        carries.append(dst)

    carry_exprs = set(c.expr for c in carries)
    idx = ctx.fresh("_s")
    with ctx.block(f"for (uint {idx} = 0; {idx} < {length}; ++{idx})"):
        outs = emit_jaxpr(ctx, body, const_vals + carries)
        # Phase 1: for each output, decide what phase 2 will copy from.
        sources = []
        for out, carry in zip(outs, carries, strict=True):
            if out.expr == carry.expr:
                sources.append(None) # self-forward: no copy needed at all
            elif out.expr in carry_exprs: # aliases a *different* carry: snapshot it
                tmp = CVal(ctx.fresh(), out.shape, out.ctype)
                ctx.emit(f"{tmp.ctype} {tmp.expr}[{tmp.size}];" if tmp.shape
                        else f"{tmp.ctype} {tmp.expr};")
                ctx.copy(tmp, out, out.size)
                sources.append(tmp)
            else:
                sources.append(out) # fresh SSA name: can't be clobbered

        # Phase 2: now it's safe to overwrite the carries.
        for src, carry in zip(sources, carries, strict=True):
            if src is not None:
                ctx.copy(carry, src, carry.size)
