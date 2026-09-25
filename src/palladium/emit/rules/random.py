"""Lowerings for one part of the MSL execution model."""

from __future__ import annotations

from jax.extend.core import JaxprEqn

from palladium.emit.core import (
    Cursor,
    CVal,
    EmitError,
    Environment,
    declare,
    rule,
)

# Threefry-2x32-20, the algorithm `jax._src.random.threefry2x32` runs by default.
_THREEFRY_ROT0 = (13, 15, 26, 6)
_THREEFRY_ROT1 = (17, 29, 16, 24)


def _emit_threefry2x32(cursor: Cursor, k1: str, k2: str, x1: str, x2: str) -> tuple[str, str]:
    """Emit uint x0, x1 = threefry2x32(k1, k2, x1, x2); return their names.

    Pure text emission, no bindings: takes a Cursor, not an Environment.
    """
    ks2 = cursor.fresh("_ks2")
    cursor.emit(f"uint {ks2} = {k1} ^ {k2} ^ 0x1BD11BDAu;")
    ks = (k1, k2, ks2)

    x0 = cursor.fresh("_tx0")
    x1n = cursor.fresh("_tx1")
    cursor.emit(f"uint {x0} = {x1} + {ks[0]};")
    cursor.emit(f"uint {x1n} = {x2} + {ks[1]};")

    def apply_round(rot: int) -> None:
        cursor.emit(f"{x0} = {x0} + {x1n};")
        cursor.emit(f"{x1n} = ({x1n} << {rot}u) | ({x1n} >> {32 - rot}u);")
        cursor.emit(f"{x1n} = {x0} ^ {x1n};")

    # 5 groups of 4 rounds, alternating rotation sets; a, b index into ks
    # for the post-group key-schedule addition, n is that round's counter.
    schedule = (
        (_THREEFRY_ROT0, 1, 2, 1),
        (_THREEFRY_ROT1, 2, 0, 2),
        (_THREEFRY_ROT0, 0, 1, 3),
        (_THREEFRY_ROT1, 1, 2, 4),
        (_THREEFRY_ROT0, 2, 0, 5),
    )
    for rots, a, b, n in schedule:
        for r in rots:
            apply_round(r)
        cursor.emit(f"{x0} = {x0} + {ks[a]};")
        cursor.emit(f"{x1n} = {x1n} + {ks[b]} + {n}u;")

    return x0, x1n


@rule("random_bits")
def _rule_random_bits(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jax.random.bits`: Threefry-2x32-20 counter-based bits, one hash
    per output element. Counter is the element's flat row-major index
    (hi word 0, lo word the index); matches jax's own construction for
    any shape under 2**32 elements, which is every real kernel output.
    """
    if eqn.params["bit_width"] != 32:
        raise EmitError("random_bits: only bit_width=32 is implemented")

    key = env.val(eqn.invars[0])
    dst = declare(env, cursor, eqn.outvars[0])
    k1, k2 = key.at("0"), key.at("1")

    with cursor.loop(dst.size) as idx:
        b0, b1 = _emit_threefry2x32(cursor, k1, k2, "0u", idx)
        cursor.emit(f"{dst.at(idx)} = {b0} ^ {b1};")


@rule("random_fold_in")
def _rule_random_fold_in(env: Environment, cursor: Cursor, eqn: JaxprEqn) -> None:
    """`jax.random.fold_in`: a fresh key from `(key, data)`, the same
    Threefry-2x32-20 hash seeded with (0, data) in place of a counter.
    Matches `jax.random.fold_in`'s `key_data` output.
    """
    key = env.val(eqn.invars[0])
    data = env.val(eqn.invars[1])
    k1, k2 = key.at("0"), key.at("1")
    b0, b1 = _emit_threefry2x32(cursor, k1, k2, "0u", data.expr)

    name = cursor.fresh()
    cursor.emit(f"uint {name}[2] = {{{b0}, {b1}}};")
    env.bind(eqn.outvars[0], CVal(expr=name, shape=(2,), ctype="uint"))
