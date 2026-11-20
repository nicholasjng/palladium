"""StableHLO programs to Metal kernels, for the `metal` PJRT plugin.

The plugin's PJRT_Client_Compile passes the portable artifact jax serializes to
`compile_program`, which parses it with jaxlib's MLIR bindings and returns a
`Program`: buffer slots and kernel launches, which `palladium._pjrt` hands to
the plugin. A program may hold elementwise and index-mapping ops
(broadcast, reshape, transpose, slices, concatenate, pad, reverse, iota), fused
into one kernel per output shape. Single-result reductions end a fusion and
materialize a buffer for subsequent kernels. Floating dots reuse Palladium's
Pallas GEMM emitter after layout normalization. Palladium kernels
(`pallas_call` lowered for `metal`, `metal_call`, `plmtl`) arrive as
`palladium_dispatch` custom calls and launch their own MSL. Kernels bind at
most 31 buffers, Metal's limit. Unsupported ops and reducer bodies raise
`UnsupportedProgram`.

Each kernel thread computes one output element at the coordinates of its flat
index. A value is generated at the coordinates it is read at, once per
coordinate tuple, so index-mapping ops cost index arithmetic, not memory.
"""

from __future__ import annotations

import dataclasses
import sys

from jax.interpreters import mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo

from palladium.stablehlo.graph import import_graph
from palladium.stablehlo.program import Launch, Program, Slot, UnsupportedProgram
from palladium.stablehlo.schedule import compile_graph

__all__ = ["Launch", "Program", "Slot", "UnsupportedProgram", "compile_program"]


# Python frames code generation may nest per StableHLO op: a convolution is
# several nodes, and a node read through a hoist and a guard several calls.
FRAMES_PER_OP = 32


def compile_program(code: bytes) -> Program:
    """Compile a serialized StableHLO portable artifact.

    Parameters
    ----------
    code : bytes
        The program as PJRT_Client_Compile receives it.

    Returns
    -------
    Program
        Slots, launches in order, and the output slots.

    Raises
    ------
    UnsupportedProgram
        An op or dtype is not supported yet.
    """
    with mlir.make_ir_context() as ctx:
        module = stablehlo.deserialize_portable_artifact(ctx, code)
        main, parameters, results, ops = import_graph(module)
        aliases = []
        if "arg_attrs" in main.attributes:
            for i, attrs in enumerate(ir.ArrayAttr(main.attributes["arg_attrs"])):
                attrs = ir.DictAttr(attrs)
                if "tf.aliasing_output" in attrs:
                    output = ir.IntegerAttr(attrs["tf.aliasing_output"]).value
                    if (
                        not 0 <= output < len(results)
                        or parameters[i].dtype != results[output].dtype
                        or parameters[i].shape != results[output].shape
                        or output in [o for _, o in aliases]
                    ):
                        raise UnsupportedProgram("metal: incompatible input/output alias")
                    aliases.append((i, output))

    # Code generation recurses along a kernel's fused values, a few frames per
    # op, and a fusion can be as deep as the program is long. Frames are heap
    # objects, so only Python's limit bounds the depth. It is raised and
    # never lowered: another thread may be compiling at that depth.
    depth = FRAMES_PER_OP * ops + 1000
    if sys.getrecursionlimit() < depth:
        sys.setrecursionlimit(depth)
    return dataclasses.replace(compile_graph(parameters, results), aliases=tuple(aliases))
