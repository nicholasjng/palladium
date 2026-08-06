# Extending the emitter

When a kernel raises `UnsupportedPrimitiveError`, the missing lowering
rule can be registered from user code with `palladium.rule`. A rule is
a function `(env, cursor, eqn) -> None` that appends MSL lines for one
jaxpr equation: `env` is the symbol table (jaxpr Var bindings, def-use
info), `cursor` is the write position into the growing MSL text. Most
rules touch both; a few need only one (a pure aliasing rule never calls
`cursor`, a pure text-emission helper never calls `env`).

A word on stability: `rule` itself is public API, but writing a rule
means touching emitter internals (`Environment`, `Cursor`, `CVal`) that
sit outside the frozen surface and may change between minor versions.
Pin your version if you ship custom rules.

## Worked example: `floor`

`jnp.floor` stages a `floor` primitive with no rule. Registering one:

```python
import palladium
from palladium.emit.core import declare

@palladium.rule("floor")
def _rule_floor(env, cursor, eqn):
    src = env.val(eqn.invars[0])            # CVal of the input
    dst = declare(env, cursor, eqn.outvars[0])  # thread-local storage, bound
    i = cursor.fresh("_i")
    with cursor.block(f"for (uint {i} = 0; {i} < {dst.size}; ++{i})"):
        cursor.emit(f"{dst.at(i)} = floor({src.at(i)});")
```

That is the whole mechanism. The pieces:

- `env.val(atom)` resolves a jaxpr atom to a `CVal`: an MSL
  expression plus its shape, C type, and address space. Literals format
  in place.
- `declare(env, cursor, var)` emits a thread-local declaration sized
  for the output and binds it, so later equations can consume it. Use
  `env.bind(var, cval)` instead when the rule aliases existing storage
  rather than producing a new value.
- `cursor.emit(line)` appends one line at the current indentation;
  `cursor.block(header)` is a context manager for a braced block;
  `cursor.fresh(prefix)` returns a unique C identifier.
- `CVal.at(index)` indexes arrays and absorbs scalars: `expr[i]` for
  shaped values, plain `expr` for rank-0.
- `eqn.params` carries the primitive's parameters (axes, dimension
  numbers, ...); `eqn.invars`/`eqn.outvars` the operands.

Rules apply to the one-thread-per-instance model, where plain element
loops are the correct shape and divergence is free.

## Non-negotiables

Copied from this project's own validation doctrine:

1. **Diff against the oracle.** Every rule lands with a test comparing
   the GPU result to `call.interpret(...)` at f32-honest tolerances.
   The interpreter is the semantics; the rule is just a translation.
2. **Break it on purpose.** Once the test is green, sabotage the rule
   (drop a term, flip an index) and confirm the test fails. A test that
   cannot fail is decoration.
3. **Watch the state discipline.** A rule must bind every outvar
   exactly once, and must not cache pointers across equations: `swap`
   results alias device memory, and scan carries are rebound per
   iteration.
4. **Cover the shapes you claim.** If the rule handles broadcasting or
   higher ranks, test those; if it does not, reject them with an
   `EmitError` naming the case instead of emitting wrong loops.
