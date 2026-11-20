"""One step of the online softmax, for the cooperative attention lowering.

The attention recognizer accepts only kernels whose scan body is exactly this
update of each query row's running maximum, sum and weighted values over one
key tile, so the emitters write it directly:

    block_max = max(scores)                  scores scaled, future keys at -inf
    new_max   = max(old_max, block_max)
    old_scale = old_sum > 0 ? exp(old_max - new_max) : 0
    p         = exp(scores - new_max)
    new_sum   = old_scale * old_sum + sum(p)
    values    = old_scale * values           the caller then adds p @ V

Metal's `max` drops a NaN operand, but a NaN score also reaches `p` and the
sum, so the row is NaN either way.
"""

from __future__ import annotations

from palladium.emit.core import Cursor, CVal

from ._shared import UNROLL


def _old_scale(old_max: str, new_max: str, old_sum: str) -> str:
    return f"(0.0f < {old_sum}) ? exp({old_max} - {new_max}) : 0.0f"


def emit_online_softmax_simd(
    cursor: Cursor,
    scores: CVal,
    row_max: CVal,
    row_sum: CVal,
    row_scale: CVal,
    scale: str,
    *,
    rows: int,
    columns: int,
    lanes: int,
    simdgroups: int,
    causal_offsets: tuple[str, str] | None = None,
) -> None:
    """Emit one update on a score tile in threadgroup memory, rows shared out
    across SIMD groups.

    Parameters
    ----------
    scores : CVal
        The (rows, columns) score tile, overwritten with probabilities.
    row_max, row_sum : CVal
        The (rows,) running maxima and sums, updated.
    row_scale : CVal
        Receives each row's `old_scale`, for the caller to rescale the values with.
    scale : str
        MSL score-scale expression.
    lanes, simdgroups : int
        SIMD-group width and count; group `sg` takes rows `sg`, `sg + simdgroups`, ...
        and lane `lane` of it the columns `lane`, `lane + lanes`, ...
    causal_offsets : tuple of str or None, optional
        Absolute positions of the tile's first key and first query, to mask keys
        after each query.

    Notes
    -----
    Lane zero commits the row state. The caller places a barrier before other
    threads read `row_scale` or the probabilities.
    """
    per_lane = -(-columns // lanes)
    partial = columns % lanes != 0

    def column(j: int) -> str:
        return "lane" if j == 0 else f"(lane + {j * lanes})"

    with cursor.block(f"for (uint row = sg; row < {rows}; row += {simdgroups})"):
        cursor.emit(f"float s[{per_lane}];")
        cursor.emit("float lane_max = -INFINITY;")
        for j in range(per_lane):
            scaled = f"({scores.at(f'row * {columns} + {column(j)}')} * {scale})"
            guards = []
            if partial:
                guards.append(f"{column(j)} < {columns}")
            if causal_offsets is not None:
                guards.append(f"({causal_offsets[0]} + {column(j)}) <= ({causal_offsets[1]} + row)")
            if guards:
                cursor.emit(f"s[{j}] = ({' && '.join(guards)}) ? {scaled} : -INFINITY;")
            else:
                cursor.emit(f"s[{j}] = {scaled};")
            cursor.emit(f"lane_max = max(lane_max, s[{j}]);")
        cursor.emit("const float block_max = simd_max(lane_max);")
        cursor.emit(f"const float old_max = {row_max.at('row')};")
        cursor.emit(f"const float old_sum = {row_sum.at('row')};")
        cursor.emit("const float new_max = max(old_max, block_max);")
        cursor.emit(f"const float old_scale = {_old_scale('old_max', 'new_max', 'old_sum')};")
        cursor.emit("float lane_sum = 0.0f;")
        for j in range(per_lane):
            store = f"{scores.at(f'row * {columns} + {column(j)}')} = s[{j}];"
            cursor.emit(f"s[{j}] = exp((s[{j}] - new_max));")
            if partial:
                with cursor.block(f"if ({column(j)} < {columns})"):
                    cursor.emit(store)
            else:
                cursor.emit(store)
            cursor.emit(f"lane_sum = (lane_sum + s[{j}]);")
        cursor.emit("const float block_sum = simd_sum(lane_sum);")
        with cursor.block("if (lane == 0)"):
            cursor.emit(f"{row_sum.at('row')} = ((old_scale * old_sum) + block_sum);")
            cursor.emit(f"{row_max.at('row')} = new_max;")
            cursor.emit(f"{row_scale.at('row')} = old_scale;")


def emit_online_softmax_fragments(
    cursor: Cursor,
    scale: str,
    *,
    columns: int,
    width: int,
    fragment_rows: int,
    causal: tuple[str, str, str] | None = None,
) -> None:
    """Emit one update on scores held in 8x8 register fragments.

    Parameters
    ----------
    scale : str
        MSL score-scale expression.
    columns, width : int
        Columns of the score tile and of the weighted-value accumulator.
    fragment_rows : int
        Fragment rows this SIMD group owns.
    causal : tuple of str or None, optional
        For a causal mask: a SIMD-group-uniform condition under which the block
        needs masking, then the key position of element `c` of fragment `j` and
        the query position of fragment row `i`, as MSL over `i`, `j` and `c`.

    Notes
    -----
    Expects `_s[i][j]` (scores, overwritten with probabilities) and `_o[i][d]`
    (weighted values, rescaled) as `vec<float, 2>` fragments, and per-lane row
    state `_m[i]`, `_l[i]`. A lane holds two adjacent elements of one row of each
    fragment, and the four lanes sharing that row differ in lane bits 0 and 3,
    so row reductions finish with two shuffles. No threadgroup memory is involved.
    """
    elements = (
        f"{UNROLL} for (short j = 0; j < {columns // 8}; ++j) "
        f"{UNROLL} for (short c = 0; c < 2; ++c)"
    )
    with cursor.block(f"{UNROLL} for (short i = 0; i < {fragment_rows}; ++i)"):
        with cursor.block(elements):
            cursor.emit(f"_s[i][j][c] = (_s[i][j][c] * {scale});")
        if causal is not None:
            needed, key, query = causal
            with cursor.block(f"if ({needed})"), cursor.block(elements):
                cursor.emit(f"_s[i][j][c] = ({key}) <= ({query}) ? _s[i][j][c] : -INFINITY;")
        cursor.emit("float block_max = -INFINITY;")
        with cursor.block(elements):
            cursor.emit("block_max = max(block_max, _s[i][j][c]);")
        for mask in (1, 8):
            cursor.emit(f"block_max = max(block_max, simd_shuffle_xor(block_max, {mask}));")
        cursor.emit("const float new_max = max(_m[i], block_max);")
        cursor.emit(f"const float old_scale = {_old_scale('_m[i]', 'new_max', '_l[i]')};")
        cursor.emit("float block_sum = 0.0f;")
        with cursor.block(elements):
            cursor.emit("_s[i][j][c] = exp((_s[i][j][c] - new_max));")
            cursor.emit("block_sum = (block_sum + _s[i][j][c]);")
        for mask in (1, 8):
            cursor.emit(f"block_sum = (block_sum + simd_shuffle_xor(block_sum, {mask}));")
        cursor.emit("_l[i] = ((old_scale * _l[i]) + block_sum);")
        cursor.emit("_m[i] = new_max;")
        with cursor.block(
            f"{UNROLL} for (short d = 0; d < {width // 8}; ++d) "
            f"{UNROLL} for (short c = 0; c < 2; ++c)"
        ):
            cursor.emit("_o[i][d][c] = (old_scale * _o[i][d][c]);")
