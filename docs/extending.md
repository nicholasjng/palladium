# Extending the emitter

When a kernel raises `UnsupportedPrimitiveError`, the missing lowering
rule can be registered from user code with `palladium.rule`. A rule is
a function `(state, eqn) -> None` that appends MSL lines for one jaxpr
equation.

A word on stability: `rule` itself is public API, but writing a rule
means touching emitter internals (`EmitState`, `CVal`) that sit outside
the frozen surface and may change between minor versions. Pin your
version if you ship custom rules.

## Worked example: `floor`

`jnp.floor` stages a `floor` primitive with no rule. Registering one:

```python
import palladium

@palladium.rule("floor")
def _rule_floor(state, eqn):
    src = state.val(eqn.invars[0])      # CVal of the input
    dst = state.declare(eqn.outvars[0]) # thread-local storage, bound
    i = state.fresh("_i")
    with state.block(f"for (uint {i} = 0; {i} < {dst.size}; ++{i})"):
        state.emit(f"{dst.at(i)} = floor({src.at(i)});")
```

That is the whole mechanism. The pieces:

- `state.val(atom)` resolves a jaxpr atom to a `CVal`: an MSL
  expression plus its shape, C type, and address space. Literals format
  in place.
- `state.declare(var)` emits a thread-local declaration sized for the
  output and binds it, so later equations can consume it. Use
  `state.bind(var, cval)` instead when the rule aliases existing
  storage rather than producing a new value.
- `state.emit(line)` appends one line at the current indentation;
  `state.block(header)` is a context manager for a braced block;
  `state.fresh(prefix)` returns a unique C identifier.
- `CVal.at(index)` indexes arrays and absorbs scalars: `expr[i]` for
  shaped values, plain `expr` for rank-0.
- `eqn.params` carries the primitive's parameters (axes, dimension
  numbers, ...); `eqn.invars`/`eqn.outvars` the operands.

Rules registered this way apply to the classic (one thread per
instance) model, where plain element loops are the correct shape and
divergence is free. The cooperative model has a parallel registry
(`palladium.emit.coop_rule`) with a layout taxonomy that rules must
respect; read the `emit/coop.py` module header before writing one.

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
