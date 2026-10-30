"""A float32 tiled matmul built from `simdgroup_matrix`, after MLX's steel GEMM.

On M1/M2-class GPUs, Metal 4's `matmul2d` reaches about 2.1 TFLOPS for
float32 at 4096³, while this kernel reaches about 2.5 (see
`notes/apple-gpu-kernel-recipe.md`). Each Pallas program is one threadgroup
of 2x2 SIMD groups. Per K step of 16, all threads copy contiguous runs of the
operand tiles into padded threadgroup rows; each lane then reads its own two
elements of every 8x8 fragment, SIMD groups interleave across the output
tile, and products run in serpentine order into per-lane float32
accumulators. Every fragment loop is fully unrolled: otherwise the
accumulator arrays live in stack memory and the kernel runs about 40x slower.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator

from palladium.emit.core import Cursor

from ._shared import SIMDGROUPS

K_TILE = 16
_WM, _WN = 2, 2
_PAD = 4  # 16 bytes of float32, as MLX pads threadgroup rows
_UNROLL = '_Pragma("clang loop unroll(full)")'
# Accumulator elements per tile above which registers spill (128x128
# measured slower than 128x64 on an M2).
_MAX_TILE_ELEMENTS = 128 * 64

_MMA = """typedef vec<float, 2> simdgroup_frag;

METAL_FUNC void simdgroup_mma(thread simdgroup_frag& d, thread simdgroup_frag& a,
                              thread simdgroup_frag& b) {
    simdgroup_matrix<float, 8, 8> am, bm, dm;
    reinterpret_cast<thread simdgroup_frag&>(am.thread_elements()) = a;
    reinterpret_cast<thread simdgroup_frag&>(bm.thread_elements()) = b;
    reinterpret_cast<thread simdgroup_frag&>(dm.thread_elements()) = d;
    simdgroup_multiply_accumulate(dm, am, bm, dm);
    d = reinterpret_cast<thread simdgroup_frag&>(dm.thread_elements());
}
"""

PARAMETERS = (
    "uint _sg [[simdgroup_index_in_threadgroup]]",
    "uint _lane [[thread_index_in_simdgroup]]",
    "uint _tid [[thread_index_in_threadgroup]]",
)


def supports(tm: int, tn: int, k: int) -> bool:
    """Whether the kernel can compute a `tm x tn` output tile over `k`.

    Parameters
    ----------
    tm, tn : int
        Output tile rows and columns.
    k : int
        Contraction length.

    Returns
    -------
    bool
        True for 2x2 SIMD groups, `tm` in 16, 32, 64 or 128, `tn` a multiple of
        16, at most 8192 tile elements, and `k` a multiple of 16.

    Notes
    -----
    The `tm` choices make each thread's share of an A tile copy divide the
    16-element row; larger tiles spill the accumulators.
    """
    return (
        SIMDGROUPS == _WM * _WN
        and tm in (16, 32, 64, 128)
        and tn % 16 == 0
        and tm * tn <= _MAX_TILE_ELEMENTS
        and k > 0
        and k % K_TILE == 0
    )


def emit_main_loop(cursor: Cursor, *, a: str, b: str, k: int, n: int, tm: int, tn: int) -> None:
    """Emit the accumulation of one output tile into per-lane fragments.

    Parameters
    ----------
    cursor : Cursor
        Destination for statements, helpers and threadgroup storage.
    a : str
        Pointer to the first row of this program's `tm x k` A strip, row stride `k`.
    b : str
        Pointer to the first column of this program's `k x tn` B strip, row stride `n`.
    k, n : int
        Contraction length and the row stride of B.
    tm, tn : int
        Output tile rows and columns; `supports(tm, tn, k)` must hold.

    Notes
    -----
    Declares `_acc[tm / 16][tn / 16]` (two float32 values per lane each) and the
    lane's fragment origin `_sm`, `_sn`; `accumulator_elements` walks them.
    Requires the parameters in `PARAMETERS`.
    """
    cursor.require("simdgroup_mma", _MMA)
    threads = SIMDGROUPS * 32
    frags_m, frags_n = tm // (8 * _WM), tn // (8 * _WN)
    lda, ldb = K_TILE + _PAD, tn + _PAD
    a_reads, b_reads = tm * K_TILE // threads, K_TILE * tn // threads
    a_cols, b_cols = K_TILE // a_reads, tn // b_reads
    a_rows, b_rows = threads // a_cols, threads // b_cols
    a_tile = cursor.allocate("float", (tm * lda,), name="_As", space="threadgroup")
    b_tile = cursor.allocate("float", (K_TILE * ldb,), name="_Bs", space="threadgroup")
    u = _UNROLL

    cursor.emit("// Each thread copies contiguous runs of both operand tiles per K step.")
    cursor.emit(f"const short _a_i = _tid / {a_cols}, _a_j = {a_reads} * (_tid % {a_cols});")
    cursor.emit(f"const short _b_i = _tid / {b_cols}, _b_j = {b_reads} * (_tid % {b_cols});")
    cursor.emit(f"device const float* _a_src = {a} + _a_i * {k} + _a_j;")
    cursor.emit(f"device const float* _b_src = {b} + _b_i * {n} + _b_j;")
    cursor.emit(f"threadgroup float* _a_dst = {a_tile.expr} + _a_i * {lda} + _a_j;")
    cursor.emit(f"threadgroup float* _b_dst = {b_tile.expr} + _b_i * {ldb} + _b_j;")
    cursor.emit("// This lane's two elements of an 8x8 fragment, as MLX's BaseMMAFrag places them.")
    cursor.emit("const short _qid = _lane / 4;")
    cursor.emit("const short _fm = (_qid & 4) + ((_lane / 2) % 4);")
    cursor.emit("const short _fn = (_qid & 2) * 2 + (_lane % 2) * 2;")
    cursor.emit(f"const short _sm = 8 * (_sg / {_WN}) + _fm, _sn = 8 * (_sg % {_WN}) + _fn;")
    cursor.emit(f"simdgroup_frag _acc[{frags_m}][{frags_n}];")
    with (
        cursor.block(f"{u} for (short i = 0; i < {frags_m}; ++i)"),
        cursor.block(f"{u} for (short j = 0; j < {frags_n}; ++j)"),
    ):
        cursor.emit("_acc[i][j] = simdgroup_frag(0);")

    with cursor.block(f"for (uint _k0 = 0; _k0 < {k}; _k0 += {K_TILE})"):
        cursor.barrier()
        with (
            cursor.block(f"{u} for (short i = 0; i < {tm}; i += {a_rows})"),
            cursor.block(f"{u} for (short j = 0; j < {a_reads}; ++j)"),
        ):
            cursor.emit(f"_a_dst[i * {lda} + j] = _a_src[i * {k} + j];")
        with (
            cursor.block(f"{u} for (short i = 0; i < {K_TILE}; i += {b_rows})"),
            cursor.block(f"{u} for (short j = 0; j < {b_reads}; ++j)"),
        ):
            cursor.emit(f"_b_dst[i * {ldb} + j] = _b_src[i * {n} + j];")
        cursor.barrier()
        cursor.emit(f"_a_src += {K_TILE};")
        cursor.emit(f"_b_src += {K_TILE * n};")
        cursor.emit(f"const threadgroup float* _as = {a_tile.expr} + _sm * {lda} + _fn;")
        cursor.emit(f"const threadgroup float* _bs = {b_tile.expr} + _fm * {ldb} + _sn;")
        with cursor.block(f"{u} for (short kk = 0; kk < {K_TILE}; kk += 8)"):
            cursor.emit(f"simdgroup_frag _a[{frags_m}], _b[{frags_n}];")
            cursor.emit("simdgroup_barrier(mem_flags::mem_none);")
            with cursor.block(f"{u} for (short i = 0; i < {frags_m}; ++i)"):
                row = f"i * {8 * _WM * lda} + kk"
                cursor.emit(f"_a[i] = simdgroup_frag(_as[{row}], _as[{row} + 1]);")
            cursor.emit("simdgroup_barrier(mem_flags::mem_none);")
            with cursor.block(f"{u} for (short j = 0; j < {frags_n}; ++j)"):
                column = f"kk * {ldb} + j * {8 * _WN}"
                cursor.emit(f"_b[j] = simdgroup_frag(_bs[{column}], _bs[{column} + 1]);")
            cursor.emit("simdgroup_barrier(mem_flags::mem_none);")
            cursor.emit("// Serpentine order keeps one operand fragment live between products.")
            with (
                cursor.block(f"{u} for (short i = 0; i < {frags_m}; ++i)"),
                cursor.block(f"{u} for (short j = 0; j < {frags_n}; ++j)"),
            ):
                cursor.emit(f"const short c = (i % 2) ? ({frags_n} - 1 - j) : j;")
                cursor.emit("simdgroup_mma(_acc[i][c], _a[i], _b[c]);")


@contextlib.contextmanager
def accumulator_elements(cursor: Cursor, tm: int, tn: int) -> Generator[tuple[str, str, str]]:
    """Visit every accumulator element this lane holds.

    Parameters
    ----------
    cursor : Cursor
        Destination positioned after `emit_main_loop`.
    tm, tn : int
        The tile passed to `emit_main_loop`.

    Yields
    ------
    value, row, column : str
        The float32 element and its row and column in the output tile, valid
        inside the `with` body.
    """
    u = _UNROLL
    with (
        cursor.block(f"{u} for (short i = 0; i < {tm // (8 * _WM)}; ++i)"),
        cursor.block(f"{u} for (short j = 0; j < {tn // (8 * _WN)}; ++j)"),
        cursor.block(f"{u} for (short e = 0; e < 2; ++e)"),
    ):
        cursor.emit(f"const uint row = _sm + i * {8 * _WM};")
        cursor.emit(f"const uint column = _sn + j * {8 * _WN} + e;")
        yield "_acc[i][j][e]", "row", "column"
