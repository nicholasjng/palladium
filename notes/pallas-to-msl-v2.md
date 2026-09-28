# Pallas-to-MSL v2

## Design decisions

- Keep the Pallas grid and its `BlockSpec` mappings as logical program
  semantics. Select execution scope separately: one MSL thread per program for
  existing scalar kernels, or one threadgroup per program for cooperative work.
- Preserve jaxpr equations, loops, and effects through planning. Lower
  primitives compositionally; Pallas-authored schedules such as online
  softmax remain explicit and do not need an attention-pattern matcher.
- Give values a logical shape, address space, and ownership distribution.
  Layout assignment is a later pass; inputs and outputs begin as device views,
  and scratch uses its declared thread or threadgroup space.
- Keep Metal-specific operations in target lowering: TensorOps calls,
  cooperative ownership, barriers, threadgroup limits, and MSL emission.
  Retain the current specialized emitter as a reference and fallback while the
  general cooperative path matures.

## First implementation slice

`src/palladium/emit/v2/plan.py` defines the launch-scope and value-layout
contract. It deliberately does not route runtime compilation through v2 yet.
Next stages are jaxpr-to-structured-IR import, layout/ownership assignment,
cooperative primitive lowering, and then routing an end-to-end Pallas kernel
through the MSL backend.

## Performance comparison

Compare v2 and the current attention emitter on the same device, shapes, and
tiles, with resident buffers and identical correctness tolerances. The existing
reference is 11.51 ms for the noncausal `[1, 4096, 4, 64]` case at `32 x 64`;
the current tile sweep's fastest tested configuration is 9.44 ms at `16 x 64`.
Report compilation separately. The goal is first to match the specialized
kernel's memory behavior and TensorOps schedule; any uplift from layout choice
or loop pipelining must be measured rather than inferred from the IR change.
