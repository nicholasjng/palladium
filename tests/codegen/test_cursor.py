"""Reusable typed emission helpers."""

import pytest

from palladium.emit.core import Cursor
from palladium.errors import EmitError


def test_cursor_allocates_typed_threadgroup_storage():
    cursor = Cursor()
    value = cursor.allocate("float", (2, 3), name="shared", space="threadgroup")

    assert value.shape == (2, 3)
    assert value.ctype == "float"
    assert value.space == "threadgroup"
    assert cursor.threadgroup_bytes == 24
    assert cursor.lines == ["    threadgroup float shared[6];"]


def test_cursor_emits_strided_loop_and_barrier():
    cursor = Cursor()
    with cursor.strided_loop("tid", "N", "THREADS", name="i") as index:
        cursor.emit(f"out[{index}] = 0.0f;")
    cursor.barrier()

    assert cursor.lines == [
        "    for (uint i = tid; i < N; i += THREADS) {",
        "        out[i] = 0.0f;",
        "    }",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
    ]


def test_cursor_rejects_zero_storage_dimensions():
    with pytest.raises(EmitError, match="dimensions must be positive"):
        Cursor().allocate("float", (0,), space="threadgroup")
