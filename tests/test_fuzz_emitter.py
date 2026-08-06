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
from hypothesis import given, settings
from hypothesis import strategies as st
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
    fn = UNARY.get(tree.op) or BINARY[tree.op]
    return fn(*(eval_tree(a, vals) for a in tree.args))


def trees(n_inputs: int):
    literals = st.floats(-2.0, 2.0, allow_nan=False, width=32).map(
        lambda v: Leaf(ref=None, lit=round(float(v), 3))
    )
    refs = st.integers(0, n_inputs - 1).map(lambda i: Leaf(ref=i, lit=None))
    return st.recursive(
        st.one_of(refs, literals),
        lambda kids: st.one_of(
            st.tuples(st.sampled_from(sorted(UNARY)), kids).map(
                lambda t: Node(t[0], (t[1],))
            ),
            st.tuples(st.sampled_from(sorted(BINARY)), kids, kids).map(
                lambda t: Node(t[0], (t[1], t[2]))
            ),
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
                    trees(n_carry + n_consts),
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
