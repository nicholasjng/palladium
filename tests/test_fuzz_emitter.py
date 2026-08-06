"""Differential fuzzing: random kernels, Metal vs the interpret oracle.

Hypothesis composes kernels from the emitter's own vocabulary (expression
trees over refs and literals; fori_loops with consts, tuple carries, and
pass-through/permuted outputs) and asserts the Metal result matches
`interpret=True`. Emitter bugs are feature *interactions* (the paren bug
hid behind an untested cast, the carry-permutation bug behind an untested
pass-through), and random composition explores exactly the programs nobody
thought to write.

Design decisions, deliberate:

- Kernels compile under MathMode.SAFE: we are testing the emitter, not
  Metal's fast math, so reassociation is removed from the comparison.
  Tolerances (1e-4) still absorb libm implementation differences between
  Metal and CPU transcendentals; emitter bugs are catastrophically wrong,
  not ulp-wrong.
- Generated ops are numerically tamed by construction (exp only as
  exp(-|x|), division by |b|+1, loop bodies wrapped in tanh, inputs in
  [-2, 2]) so failures mean wrong code, not overflow; false-positive
  discipline is what keeps a fuzzer worth running.
- GPU-bound and slow, so opt-in: `uv run pytest -m fuzz`. Scale with
  PALLADIUM_FUZZ_EXAMPLES (default 100 per test). On failure Hypothesis
  shrinks to a minimal kernel and prints a @reproduce_failure blob; turn
  the shrunk case into a named regression test, like test_carry_permutation.
"""

import dataclasses
import os

import jax
import jax.numpy as jnp
import metal_runtime as mr
import numpy as np
import pytest
from hypothesis import assume as hyp_assume, given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from jax.experimental import pallas as pl

import palladium

pytestmark = pytest.mark.fuzz

MAX_EXAMPLES = int(os.environ.get("PALLADIUM_FUZZ_EXAMPLES", "100"))
N = 16  # array length for every operand; small keeps compiles snappy

UNARY = {
    "neg": lambda a: -a,
    "abs": jnp.abs,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tanh": jnp.tanh,
    "sqrt_abs": lambda a: jnp.sqrt(jnp.abs(a)),
    "exp_negabs": lambda a: jnp.exp(-jnp.abs(a)),  # bounded (0, 1]
}
BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "safe_div": lambda a, b: a / (jnp.abs(b) + 1.0),
    "min": jnp.minimum,
    "max": jnp.maximum,
}
# Comparisons appear only inside where-nodes, and only over *leaves*: raw
# inputs and literals are bit-identical on both backends, so the predicate
# cannot flip on ulp noise between Metal and CPU libm. Computed comparison
# operands would turn last-ulp differences into branch divergence, a false
# positive no tolerance absorbs.
COMPARE = {
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
}


@dataclasses.dataclass(frozen=True)
class Leaf:
    ref: int | None  # read input `ref`, or...
    lit: float | None  # ...an inline literal


@dataclasses.dataclass(frozen=True)
class Node:
    op: str
    args: tuple


def eval_tree(tree, vals):
    if isinstance(tree, Leaf):
        return vals[tree.ref] if tree.ref is not None else tree.lit
    if tree.op.startswith("where_"):
        lhs, rhs, on_true, on_false = (eval_tree(a, vals) for a in tree.args)
        pred = COMPARE[tree.op.removeprefix("where_")](lhs, rhs)
        return jnp.where(pred, on_true, on_false)
    fn = UNARY.get(tree.op) or BINARY[tree.op]
    return fn(*(eval_tree(a, vals) for a in tree.args))


def trees(n_inputs: int, compare_refs: range | None = None):
    literals = st.floats(-2.0, 2.0, allow_nan=False, width=32).map(
        lambda v: Leaf(ref=None, lit=round(float(v), 3))
    )
    refs = st.integers(0, n_inputs - 1).map(lambda i: Leaf(ref=i, lit=None))
    leaves = st.one_of(refs, literals)
    # Comparison operands must be bit-identical on both backends (see
    # COMPARE). In loop bodies the tree's ref leaves are *carries*, i.e.
    # computed values after the first iteration, so callers there pass
    # compare_refs restricted to the loop consts; a comparison against a
    # carry once flipped a branch on a 1-ulp tanh difference.
    if compare_refs is None:
        cmp_leaves = leaves
    elif len(compare_refs) > 0:
        cmp_refs = st.sampled_from(list(compare_refs)).map(
            lambda i: Leaf(ref=i, lit=None)
        )
        cmp_leaves = st.one_of(cmp_refs, literals)
    else:
        cmp_leaves = literals
    return st.recursive(
        leaves,
        lambda kids: st.one_of(
            st.tuples(st.sampled_from(sorted(UNARY)), kids).map(
                lambda t: Node(t[0], (t[1],))
            ),
            st.tuples(st.sampled_from(sorted(BINARY)), kids, kids).map(
                lambda t: Node(t[0], (t[1], t[2]))
            ),
            # Conditionals: comparison over stable leaves only, general
            # branches. Every instance crosses jit-inlining + select_n.
            st.tuples(
                st.sampled_from(sorted(COMPARE)), cmp_leaves, cmp_leaves, kids, kids
            ).map(lambda t: Node(f"where_{t[0]}", (t[1], t[2], t[3], t[4]))),
        ),
        max_leaves=8,
    )


def input_arrays(count: int):
    return st.tuples(
        *[
            arrays(np.float32, (N,), elements=st.floats(-2.0, 2.0, width=32))
            for _ in range(count)
        ]
    )


def _assert_matches_oracle(f, args):
    got = f(*args)
    want = f.interpret(*args)
    got = got if isinstance(got, tuple) else (got,)
    want = want if isinstance(want, tuple) else (want,)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_allclose(g, np.asarray(w), rtol=1e-4, atol=1e-4)


@st.composite
def elementwise_cases(draw):
    n_inputs = draw(st.integers(1, 3))
    tree = draw(trees(n_inputs))
    blocked = draw(st.booleans())  # gridless vs 1D grid over blocks of 8
    args = draw(input_arrays(n_inputs))
    return n_inputs, tree, blocked, args


def make_elementwise_kernel(tree, n_inputs):
    def kernel(*refs):
        *ins, out = refs
        vals = [r[...] for r in ins]
        res = eval_tree(tree, vals)
        if jnp.shape(res) != jnp.shape(vals[0]):
            # Literal-only tree folded to a scalar: anchor it to an input's
            # shape so the store sees a block (stages mul/add, both covered).
            res = res + 0.0 * vals[0]
        out[...] = res

    return kernel


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(elementwise_cases())
def test_fuzz_elementwise(case):
    n_inputs, tree, blocked, args = case
    kwargs: dict = {"out_shape": jax.ShapeDtypeStruct((N,), jnp.float32)}
    if blocked:
        spec = pl.BlockSpec((8,), lambda i: (i,))
        kwargs |= {"grid": (N // 8,), "in_specs": [spec] * n_inputs, "out_specs": spec}
    f = palladium.metal_call(
        make_elementwise_kernel(tree, n_inputs),
        math_mode=mr.MathMode.SAFE,
        **kwargs,
    )
    _assert_matches_oracle(f, args)


@st.composite
def loop_cases(draw):
    n_carry = draw(st.integers(1, 3))
    n_consts = draw(st.integers(0, 2))
    length = draw(st.integers(1, 7))
    # Per carry: either pass another carry through unchanged (the aliasing
    # cases: self-forward and permutation) or compute a tamed expression
    # over all carries and consts.
    plans = draw(
        st.tuples(
            *[
                st.one_of(
                    st.integers(0, n_carry - 1),  # pass-through of carry j
                    trees(
                        n_carry + n_consts,
                        compare_refs=range(n_carry, n_carry + n_consts),
                    ),
                )
                for _ in range(n_carry)
            ]
        )
    )
    args = draw(input_arrays(n_carry + n_consts))
    return n_carry, n_consts, length, plans, args


def make_loop_kernel(n_carry, n_consts, length, plans):
    def kernel(*refs):
        ins, outs = refs[: n_carry + n_consts], refs[n_carry + n_consts :]
        init = tuple(r[...] for r in ins[:n_carry])
        consts = [r[...] for r in ins[n_carry:]]

        def body(_, carry):
            vals = list(carry) + consts
            new = []
            for plan in plans:
                if isinstance(plan, int):
                    new.append(carry[plan])
                else:
                    # tanh bounds the carry so `length` iterations cannot
                    # overflow: divergence then means wrong code, not
                    # accumulated extremes.
                    new.append(jnp.tanh(eval_tree(plan, vals) + 0.0 * carry[0]))
            return tuple(new)

        final = jax.lax.fori_loop(0, length, body, init)
        for out, value in zip(outs, final):
            out[...] = value

    return kernel


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(loop_cases())
def test_fuzz_loops(case):
    n_carry, n_consts, length, plans, args = case
    out_shape = tuple(jax.ShapeDtypeStruct((N,), jnp.float32) for _ in range(n_carry))
    f = palladium.metal_call(
        make_loop_kernel(n_carry, n_consts, length, plans),
        math_mode=mr.MathMode.SAFE,
        out_shape=out_shape,
    )
    _assert_matches_oracle(f, args)


@st.composite
def switch_cases(draw):
    n_inputs = draw(st.integers(1, 3))
    n_branches = draw(st.integers(2, 4))
    branches = tuple(draw(trees(n_inputs)) for _ in range(n_branches))
    blocked = draw(st.booleans())
    args = draw(input_arrays(n_inputs))
    return n_inputs, branches, blocked, args


def make_switch_kernel(branches, n_inputs):
    # The branch index derives from abs+scale+cast, all exact float ops,
    # so it is bit-identical on both backends (same reasoning as COMPARE:
    # no computed float compare may sit under control flow). Indexed off
    # the ref, not the loaded array: value-level indexing stages `slice`,
    # which the emitter does not lower.
    def kernel(*refs):
        *ins, out = refs
        vals = [r[...] for r in ins]
        idx = (jnp.abs(ins[0][0]) * 2.0).astype(jnp.int32)  # 0..4, clamped
        res = jax.lax.switch(
            idx,
            [
                lambda tree=tree: eval_tree(tree, vals) + 0.0 * vals[0]
                for tree in branches
            ],
        )
        out[...] = res

    return kernel


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(switch_cases())
def test_fuzz_cond(case):
    """`lax.cond`/`lax.switch`: divergent branch selection per program
    instance (blocked grids give each block its own index), clamped
    switch semantics, branch bodies from the same tree grammar."""
    n_inputs, branches, blocked, args = case
    kwargs: dict = {"out_shape": jax.ShapeDtypeStruct((N,), jnp.float32)}
    if blocked:
        spec = pl.BlockSpec((8,), lambda i: (i,))
        kwargs |= {"grid": (N // 8,), "in_specs": [spec] * n_inputs, "out_specs": spec}
    f = palladium.metal_call(
        make_switch_kernel(branches, n_inputs),
        math_mode=mr.MathMode.SAFE,
        **kwargs,
    )
    _assert_matches_oracle(f, args)


@st.composite
def while_cases(draw):
    n_carry = draw(st.integers(1, 2))
    n_consts = draw(st.integers(0, 1))
    plans = draw(
        st.tuples(
            *[
                st.one_of(
                    st.integers(0, n_carry - 1),
                    trees(
                        n_carry + n_consts,
                        compare_refs=range(n_carry, n_carry + n_consts),
                    ),
                )
                for _ in range(n_carry)
            ]
        )
    )
    blocked = draw(st.booleans())
    args = draw(input_arrays(n_carry + n_consts))
    return n_carry, n_consts, plans, blocked, args


def make_while_kernel(n_carry, n_consts, plans):
    # Trip count 0..4 derives from exact float ops on input element 0, so
    # every instance's count is bit-identical on both backends while
    # *differing across instances* under a blocked grid: divergent
    # while-loop trip counts are exactly what the classic model must
    # support.
    def kernel(*refs):
        ins, outs = refs[: n_carry + n_consts], refs[n_carry + n_consts :]
        init = tuple(r[...] for r in ins[:n_carry])
        consts = [r[...] for r in ins[n_carry:]]
        trips = (jnp.abs(ins[0][0]) * 2.0).astype(jnp.int32)

        def cond(state):
            return state[0] < trips

        def body(state):
            i, *carry = state
            vals = list(carry) + consts
            new = []
            for plan in plans:
                if isinstance(plan, int):
                    new.append(carry[plan])
                else:
                    new.append(jnp.tanh(eval_tree(plan, vals) + 0.0 * carry[0]))
            return (i + 1, *new)

        final = jax.lax.while_loop(cond, body, (jnp.int32(0), *init))
        for out, value in zip(outs, final[1:]):
            out[...] = value

    return kernel


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(while_cases())
def test_fuzz_while(case):
    n_carry, n_consts, plans, blocked, args = case
    block = 8
    out_shape = tuple(jax.ShapeDtypeStruct((N,), jnp.float32) for _ in range(n_carry))
    kwargs: dict = {"out_shape": out_shape}
    if blocked:
        spec = pl.BlockSpec((block,), lambda i: (i,))
        kwargs |= {
            "grid": (N // block,),
            "in_specs": [spec] * (n_carry + n_consts),
            "out_specs": [spec] * n_carry,
        }
    f = palladium.metal_call(
        make_while_kernel(n_carry, n_consts, plans),
        math_mode=mr.MathMode.SAFE,
        **kwargs,
    )
    _assert_matches_oracle(f, args)


@st.composite
def coop_dot_cases(draw):
    rows = draw(st.sampled_from([1, 2, 4, 8, 16, 32]))
    k = draw(st.sampled_from([8, 16, 32, 64]))
    n = draw(st.sampled_from([8, 16, 32, 64, 128, 256]))
    transpose = draw(st.booleans())
    # Over the MMA threadgroup-tile budget the kernel falls back to the
    # classic model, whose per-thread arrays then exceed the stack and
    # reject cleanly; not a useful fuzz target.
    hyp_assume(rows * n * 4 <= 16384)
    epilogue = draw(st.sampled_from(["none", "tanh", "scale", "rowsum"]))
    seed = draw(st.integers(0, 2**31 - 1))
    return rows, k, n, transpose, epilogue, seed


@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(coop_dot_cases())
def test_fuzz_cooperative_dots(case):
    """Dot-anchored kernels across the cooperative eligibility space:
    varying R (rows), k, n hits the MMA path, the float4 rowdot path,
    and the generic sharded path; epilogues cross the dot's threadgroup
    or sharded output into elementwise and reduction rules."""
    rows, k, n, transpose, epilogue, seed = case
    rng = np.random.default_rng(seed)
    q = rng.uniform(-1, 1, (rows, k)).astype(np.float32)
    kv = rng.uniform(-1, 1, (n, k) if transpose else (k, n)).astype(np.float32)

    out_cols = 1 if epilogue == "rowsum" else n

    def kernel(q_ref, k_ref, o_ref):
        kk = k_ref[...]
        s = jnp.dot(q_ref[...], kk.T if transpose else kk)
        if epilogue == "tanh":
            s = jnp.tanh(s)
        elif epilogue == "scale":
            s = s * 0.5 + 1.0
        elif epilogue == "rowsum":
            s = jnp.sum(s, axis=1, keepdims=True)
        o_ref[...] = s

    f = palladium.metal_call(
        kernel,
        math_mode=mr.MathMode.SAFE,
        out_shape=jax.ShapeDtypeStruct((rows, out_cols), jnp.float32),
    )
    _assert_matches_oracle(f, (q, kv))
