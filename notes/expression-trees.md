# Scalar expressions as pytrees (planned, not built)

The emitter builds C expressions as strings: self-bracketing templates in
`numeric.py`, `$i` substitution in lazy values, paren counting in
`unwrapped()` / `enclosed()`, and `{a}` counting for fusion. The plan is a
small expression tree, printed with C++ precedence, with the nodes
registered as JAX pytrees so `jax.tree_util` does the generic traversal.

## Nodes

Frozen dataclasses in `emit/expr.py`:

- `Atom(text)`: name, literal, or indexed storage. Not registered, so it
  is a pytree leaf.
- `Infix(op, lhs, rhs)` and `Prefix(op, arg)`: operators.
- `Call(name, args: tuple)`, `Cast(ctype, arg)`, `Select(cond, a, b)`.
- `Index()`: the flat-index placeholder that replaces `$i`.

Register each with
`jax.tree_util.register_dataclass(cls, data_fields=[operands], meta_fields=[op, name, ctype])`.
Operands are data and everything else is metadata, so the treedef carries
the operators. Never store `None` as an operand: pytrees read it as an
empty subtree.

## What tree_util provides

- Use counts: `sum(leaf is x for leaf in tree_leaves(e))` replaces the
  `template.count("{a}")` fusion check.
- Substitution: `tree_map(lambda l: idx if isinstance(l, Index) else l, e)`
  replaces the `$i` string replacement in `CVal.at`.
- Structural keys: `(tree_structure(e), tuple(tree_leaves(e)))` is
  hashable and equal only for identical expressions, so it can drive
  common-subexpression elimination.
- One level of children:
  `tree_flatten(e, is_leaf=lambda n: n is not e)` returns the root's
  children and a treedef that rebuilds it. This is enough for generic
  bottom-up rewrites such as constant folding.

Printing stays a method per node class, since it needs the node type and
its precedence. tree_util only exposes leaves and structure.

## Printing rules

- C++ precedence table. Bitwise `& ^ |` bind below comparisons.
- Only left-nested chains of equal precedence drop parentheses. Keep them
  on a right-nested `a + (b + c)`, even for `+` and `*`: float arithmetic
  is not associative, and SAFE math must keep the jaxpr's order.
- Literals from `literal()` become Atoms. Parenthesize a negative literal
  under a prefix operator (`-(-1.0f)`).

## Steps

1. Add `expr.py`: nodes, pytree registration, and the printer. Wrap
   existing strings as Atoms with `enclosed()` so everything still emits.
2. Turn the operator entries of `ELEMENTWISE` into `Infix` / `Prefix`,
   and the call entries into `Call`. Regenerate the golden snapshots and
   run the fuzz tests.
3. Rewrite `typed_expression` and the `_template` special cases (NaN-aware
   extrema, integer div and rem, shifts, `sign`, `one_minus_square`) as
   builder functions. Builders that repeat an operand bind it once.
4. Replace `CVal.lazy` strings with trees and an `Index` leaf, and move
   the fusion check to leaf counts. Remove `unwrapped()` and `enclosed()`.
5. Convert the remaining string builders: the matmul epilogue and
   softmax (`format_scalar`), the cumulative and reduction combines, and
   the random rules.
6. Fold `CExpr` in `addressing.py` into the integer subset of the same
   tree, with its constant folding as a bottom-up rewrite.
7. Optional: CSE over structural keys inside one emitted statement.

The emitted MSL stays semantically equivalent at every step; the fuzz
tests guard each one.
