"""The runtime's C API through ctypes, as an XLA FFI handler calls it: only
the C ABI in `native/runtime/c_api.h`, no Python or GIL in the calls."""

import ctypes
import mmap
import os
import threading

import numpy as np
import pytest

from palladium import _runtime as rt

_ADD_ONE_SOURCE = b"""
#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    buf[tid] = buf[tid] + 1.0f;
}
"""


@pytest.fixture(scope="module")
def lib():
    return rt.library()


def _compile(lib, source: bytes, math_mode: int = rt.MATH_MODES["fast"]):
    err = ctypes.c_char_p()
    library = ctypes.c_void_p()
    status = lib.mr_compile_library(
        source, len(source), math_mode, ctypes.byref(library), ctypes.byref(err)
    )
    return status, library, err


def _get_pipeline(lib, library, function_name: bytes):
    err = ctypes.c_char_p()
    pipeline = ctypes.c_void_p()
    status = lib.mr_get_pipeline(library, function_name, ctypes.byref(pipeline), ctypes.byref(err))
    return status, pipeline, err


def _dispatch_add_one(lib, pipeline, buffer, n: int):
    buffers_arr = (ctypes.c_void_p * 1)(buffer)
    offsets_arr = (ctypes.c_size_t * 1)(0)
    desc = rt.LaunchDesc()
    desc.pipeline = pipeline
    desc.buffers = buffers_arr
    desc.buffer_offsets = offsets_arr
    desc.buffer_count = 1
    desc.scalar_count = 0
    desc.threadgroup_memory_count = 0
    desc.grid_x, desc.grid_y, desc.grid_z = n, 1, 1
    desc.threadgroup_x, desc.threadgroup_y, desc.threadgroup_z = n, 1, 1
    err = ctypes.c_char_p()
    status = lib.mr_dispatch(ctypes.byref(desc), ctypes.byref(err))
    return status, err


def _dispatch_async(lib, desc):
    """One launch in flight: mr_batch_create, mr_batch_add, mr_batch_commit.
    On failure the batch is released and the returned handle is null."""
    batch = ctypes.c_void_p()
    err = ctypes.c_char_p()
    status = lib.mr_batch_create(ctypes.byref(batch), ctypes.byref(err))
    if status == rt.OK:
        status = lib.mr_batch_add(batch, ctypes.byref(desc), ctypes.byref(err))
    if status == rt.OK:
        status = lib.mr_batch_commit(batch, ctypes.byref(err))
    if status != rt.OK and batch.value:
        lib.mr_release_batch(batch)
        batch = ctypes.c_void_p()
    return status, batch, err


def test_device_info_reports_device_limits(lib):
    info = rt.DeviceInfo()
    err = ctypes.c_char_p()
    assert lib.mr_device_info(ctypes.byref(info), ctypes.byref(err)) == rt.OK, err.value
    assert info.max_threads_per_threadgroup >= 256
    assert info.max_threadgroup_memory_length >= 16 * 1024
    assert info.has_unified_memory
    assert info.apple_gpu_family >= 7  # Apple silicon
    assert info.name.startswith(b"Apple")

    assert lib.mr_device_info(None, ctypes.byref(err)) == rt.ERROR_INVALID_ARGUMENT
    assert err.value is not None
    assert b"out_info is null" in err.value
    lib.mr_free_error_message(err)


def test_pipeline_info_reports_the_simdgroup_width_and_kernel_limits(lib):
    source = b"""
    #include <metal_stdlib>
    using namespace metal;

    kernel void staged(device float* buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
        threadgroup float scratch[64];
        scratch[tid % 64] = buf[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[tid] = scratch[(tid + 1) % 64];
    }
    """
    status, library, err = _compile(lib, source)
    assert status == rt.OK, err.value
    status, pipeline, err = _get_pipeline(lib, library, b"staged")
    assert status == rt.OK, err.value

    info = rt.PipelineInfo()
    err = ctypes.c_char_p()
    status = lib.mr_pipeline_info(pipeline, ctypes.byref(info), ctypes.byref(err))
    assert status == rt.OK, err.value

    assert info.thread_execution_width == 32
    assert 0 < info.max_threads_per_threadgroup <= 1024
    assert info.static_threadgroup_memory_length >= 64 * 4

    status = lib.mr_pipeline_info(None, ctypes.byref(info), ctypes.byref(err))
    assert status == rt.ERROR_INVALID_ARGUMENT
    assert err.value is not None
    assert b"pipeline is null" in err.value
    lib.mr_free_error_message(err)

    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_compile_and_dispatch_round_trip_with_a_misaligned_buffer(lib):
    """A plain numpy allocation isn't page-aligned: exercises whichever
    wrap path mr_wrap_buffer picks for it (page-rounded zero-copy or, if
    that range isn't safely mappable, the owned-copy fallback); either
    way mr_buffer_flush_to must leave `array` correct."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value

    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == rt.OK, err.value

    array = np.arange(16, dtype=np.float32)
    ptr = array.ctypes.data_as(ctypes.c_void_p)

    err = ctypes.c_char_p()
    buffer = ctypes.c_void_p()
    status = lib.mr_wrap_buffer(ptr, array.nbytes, ctypes.byref(buffer), ctypes.byref(err))
    assert status == rt.OK, err.value

    status, err = _dispatch_add_one(lib, pipeline, buffer, 16)
    assert status == rt.OK, err.value

    err = ctypes.c_char_p()
    status = lib.mr_buffer_flush_to(buffer, ctypes.byref(err))
    assert status == rt.OK, err.value

    assert np.array_equal(array, np.arange(16, dtype=np.float32) + 1.0)

    lib.mr_release_buffer(buffer)
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_dispatch_writes_are_visible_without_flush_for_a_page_rounded_buffer(lib):
    """Proves the page-rounded wrap is actually zero-copy, not merely
    error-free: dispatches into a buffer with a realistic (64-byte,
    XLA-style) but non-page-aligned offset inside a page-aligned mmap
    region, then reads the array back *without* calling
    mr_buffer_flush_to. Only true memory aliasing makes that visible."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value

    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == rt.OK, err.value

    region = mmap.mmap(-1, mmap.PAGESIZE)
    base = ctypes.addressof(ctypes.c_char.from_buffer(region))
    assert base % mmap.PAGESIZE == 0
    offset = 64  # SIMD-width aligned, deliberately not page-aligned
    n = 16
    array = np.frombuffer(region, dtype=np.float32, count=n, offset=offset)
    array[:] = np.arange(n, dtype=np.float32)
    ptr = ctypes.c_void_p(base + offset)

    err = ctypes.c_char_p()
    buffer = ctypes.c_void_p()
    status = lib.mr_wrap_buffer(ptr, array.nbytes, ctypes.byref(buffer), ctypes.byref(err))
    assert status == rt.OK, err.value

    status, err = _dispatch_add_one(lib, pipeline, buffer, n)
    assert status == rt.OK, err.value

    # Deliberately no mr_buffer_flush_to call here.
    assert np.array_equal(array, np.arange(n, dtype=np.float32) + 1.0)

    lib.mr_release_buffer(buffer)
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_compile_and_dispatch_round_trip_with_a_page_aligned_buffer(lib):
    """An anonymous mmap is page-aligned by construction: exercises the
    zero-copy wrap path."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value

    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == rt.OK, err.value

    size = mmap.PAGESIZE
    region = mmap.mmap(-1, size)
    array = np.frombuffer(region, dtype=np.float32, count=16)
    array[:] = np.arange(16, dtype=np.float32)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    assert addr % mmap.PAGESIZE == 0

    err = ctypes.c_char_p()
    buffer = ctypes.c_void_p()
    status = lib.mr_wrap_buffer(
        ctypes.c_void_p(addr), size, ctypes.byref(buffer), ctypes.byref(err)
    )
    assert status == rt.OK, err.value

    status, err = _dispatch_add_one(lib, pipeline, buffer, 16)
    assert status == rt.OK, err.value

    err = ctypes.c_char_p()
    status = lib.mr_buffer_flush_to(buffer, ctypes.byref(err))
    assert status == rt.OK, err.value

    assert np.array_equal(array, np.arange(16, dtype=np.float32) + 1.0)

    lib.mr_release_buffer(buffer)
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_invalid_msl_returns_compile_error_status(lib):
    bad_source = b"this is not valid msl {{{"
    status, _, err = _compile(lib, bad_source)
    assert status == rt.ERROR_COMPILE
    assert err.value
    lib.mr_free_error_message(err)


def test_missing_function_returns_function_not_found_status(lib):
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value

    status, _, err = _get_pipeline(lib, library, b"no_such_kernel")
    assert status == rt.ERROR_FUNCTION_NOT_FOUND
    assert err.value
    lib.mr_free_error_message(err)
    lib.mr_release_library(library)


def test_non_utf8_strings_return_invalid_argument_instead_of_aborting(lib):
    """Metal asserts on the nil NSString that invalid UTF-8 converts to,
    which would abort the host process."""
    status, _, err = _compile(lib, b"#include <metal_stdlib>\n// \xff\n")
    assert status == rt.ERROR_INVALID_ARGUMENT
    assert b"MSL source is not valid UTF-8" in err.value
    lib.mr_free_error_message(err)

    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value
    status, _, err = _get_pipeline(lib, library, b"add_one\xff")
    assert status == rt.ERROR_INVALID_ARGUMENT
    assert b"function name is not valid UTF-8" in err.value
    lib.mr_free_error_message(err)
    lib.mr_release_library(library)


def test_embedded_nul_in_source_is_rejected_rather_than_truncated(lib):
    """mr_compile_library takes a length, so a NUL is part of the source."""
    status, _, err = _compile(lib, _ADD_ONE_SOURCE + b"\0trailing")
    assert status == rt.ERROR_INVALID_ARGUMENT
    assert b"NUL" in err.value
    lib.mr_free_error_message(err)


def test_repeated_compile_and_release_does_not_crash(lib):
    for _ in range(50):
        status, library, err = _compile(lib, _ADD_ONE_SOURCE)
        assert status == rt.OK, err.value
        status, pipeline, err = _get_pipeline(lib, library, b"add_one")
        assert status == rt.OK, err.value
        lib.mr_release_pipeline(pipeline)
        lib.mr_release_library(library)


def test_concurrent_dispatch_against_one_shared_pipeline_does_not_corrupt(lib):
    """XLA's CPU backend runs FFI handlers on its own compute thread pool
    (xla/ffi/api/c_api.h), with no trait to opt out of concurrent calls, so
    a real handler backing mr_dispatch has to survive many threads racing
    mr_get_pipeline on one shared MRLibrary and dispatching through the
    resulting MRPipeline at once. Each thread wraps and dispatches into its
    own buffer; correctness of the readback is what actually proves no
    shared cache/state got corrupted, not just the absence of a crash."""
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value

    n_threads = 24
    n_iters = 20
    barrier = threading.Barrier(n_threads)
    errors: list[tuple[int, str]] = []
    errors_lock = threading.Lock()

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for it in range(n_iters):
                status, pipeline, err = _get_pipeline(lib, library, b"add_one")
                assert status == rt.OK, err.value

                n = 11 + tid
                array = np.arange(n, dtype=np.float32)
                ptr = array.ctypes.data_as(ctypes.c_void_p)

                err = ctypes.c_char_p()
                buffer = ctypes.c_void_p()
                status = lib.mr_wrap_buffer(
                    ptr, array.nbytes, ctypes.byref(buffer), ctypes.byref(err)
                )
                assert status == rt.OK, err.value

                status, err = _dispatch_add_one(lib, pipeline, buffer, n)
                assert status == rt.OK, err.value

                err = ctypes.c_char_p()
                status = lib.mr_buffer_flush_to(buffer, ctypes.byref(err))
                assert status == rt.OK, err.value

                expect = np.arange(n, dtype=np.float32) + 1.0
                if not np.array_equal(array, expect):
                    raise AssertionError(f"tid={tid} it={it}: got {array[:5]}, want {expect[:5]}")

                lib.mr_release_buffer(buffer)
                lib.mr_release_pipeline(pipeline)
        except Exception as e:  # noqa: BLE001 - collected and re-raised on the main thread
            with errors_lock:
                errors.append((tid, repr(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lib.mr_release_library(library)

    assert not errors, errors


def test_async_dispatch_overlaps_and_offsets_partition_one_buffer(lib):
    """Several in-flight batches, each aimed at a different offset of the
    same wrapped buffer, must all land."""
    n_per_launch, n_launches = 256, 8
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value
    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == rt.OK, err.value

    data = np.arange(n_per_launch * n_launches, dtype=np.float32)
    buffer = ctypes.c_void_p()
    err = ctypes.c_char_p()
    status = lib.mr_wrap_buffer(
        data.ctypes.data_as(ctypes.c_void_p),
        data.nbytes,
        ctypes.byref(buffer),
        ctypes.byref(err),
    )
    assert status == rt.OK, err.value

    batches = []
    for launch_index in range(n_launches):
        buffers_arr = (ctypes.c_void_p * 1)(buffer)
        offsets_arr = (ctypes.c_size_t * 1)(launch_index * n_per_launch * 4)
        desc = rt.LaunchDesc()
        desc.pipeline = pipeline
        desc.buffers = buffers_arr
        desc.buffer_offsets = offsets_arr
        desc.buffer_count = 1
        desc.scalar_count = 0
        desc.threadgroup_memory_count = 0
        desc.grid_x, desc.grid_y, desc.grid_z = n_per_launch, 1, 1
        desc.threadgroup_x, desc.threadgroup_y, desc.threadgroup_z = 0, 0, 0
        status, batch, err = _dispatch_async(lib, desc)
        assert status == rt.OK, err.value
        batches.append(batch)

    for batch in batches:
        err = ctypes.c_char_p()
        assert lib.mr_batch_wait(batch, ctypes.byref(err)) == rt.OK, err.value
        lib.mr_release_batch(batch)

    err = ctypes.c_char_p()
    assert lib.mr_buffer_flush_to(buffer, ctypes.byref(err)) == rt.OK, err.value
    np.testing.assert_array_equal(
        data, np.arange(n_per_launch * n_launches, dtype=np.float32) + 1.0
    )

    lib.mr_release_buffer(buffer)
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


@pytest.mark.parametrize("async_dispatch", [False, True])
@pytest.mark.parametrize("offset", [4, 64, 2**64 - 64])
def test_dispatch_rejects_offsets_outside_logical_wrapping(lib, async_dispatch, offset):
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value
    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == rt.OK, err.value
    buffer = ctypes.c_void_p()
    batch = ctypes.c_void_p()
    with mmap.mmap(-1, mmap.PAGESIZE) as region:
        data = np.frombuffer(region, dtype=np.float32)
        data[:] = 10
        try:
            # The logical allocation starts inside the page. The huge offset
            # used to wrap back to the page's beginning when 64 was added.
            status = lib.mr_wrap_buffer(
                data.ctypes.data + 64, 4, ctypes.byref(buffer), ctypes.byref(err)
            )
            assert status == rt.OK, err.value
            desc = rt.LaunchDesc()
            desc.pipeline = pipeline
            desc.buffers = (ctypes.c_void_p * 1)(buffer)
            desc.buffer_offsets = (ctypes.c_size_t * 1)(offset)
            desc.buffer_count = 1
            desc.grid_x = desc.grid_y = desc.grid_z = 1
            desc.threadgroup_x = desc.threadgroup_y = desc.threadgroup_z = 1
            if async_dispatch:
                status, batch, err = _dispatch_async(lib, desc)
            else:
                status = lib.mr_dispatch(ctypes.byref(desc), ctypes.byref(err))
            assert status == 5, err.value
            assert b"logical buffer of 4 bytes" in err.value
            assert not batch.value
            np.testing.assert_array_equal(data, np.full(data.shape, 10, dtype=np.float32))
            lib.mr_free_error_message(err)
            err = ctypes.c_char_p()
            # Offset zero remains valid after the rejected launch.
            status, err = _dispatch_add_one(lib, pipeline, buffer, 1)
            assert status == rt.OK, err.value
            assert data[16] == 11
        finally:
            if batch.value:
                lib.mr_batch_wait(batch, ctypes.byref(err))
                lib.mr_release_batch(batch)
            lib.mr_release_buffer(buffer)
            lib.mr_release_pipeline(pipeline)
            lib.mr_release_library(library)
            del data


@pytest.mark.parametrize(
    "ptr, size", [(1, 2**64 - 1), (2**64 - 16, 32), (2**64 - mmap.PAGESIZE, 1)]
)
def test_wrap_rejects_unrepresentable_memory_ranges(lib, ptr, size):
    buffer = ctypes.c_void_p()
    err = ctypes.c_char_p()
    status = lib.mr_wrap_buffer(ptr, size, ctypes.byref(buffer), ctypes.byref(err))
    assert status == 5, err.value
    assert not buffer.value
    assert err.value and b"overflows" in err.value
    lib.mr_free_error_message(err)


def test_zero_length_wrap_still_accepts_zero_offset(lib):
    source = b"kernel void noop() {}"
    status, library, err = _compile(lib, source)
    assert status == rt.OK, err.value
    status, pipeline, err = _get_pipeline(lib, library, b"noop")
    assert status == rt.OK, err.value
    buffer = ctypes.c_void_p()
    assert lib.mr_wrap_buffer(None, 0, ctypes.byref(buffer), ctypes.byref(err)) == rt.OK
    try:
        status, err = _dispatch_add_one(lib, pipeline, buffer, 1)
        assert status == rt.OK, err.value
    finally:
        lib.mr_release_buffer(buffer)
        lib.mr_release_pipeline(pipeline)
        lib.mr_release_library(library)


def test_batch_add_encodes_many_launches_into_one_command_buffer(lib):
    """mr_batch_create/add/commit/wait: the descriptor is copied on add, so
    one descriptor whose offsets are rewritten between adds partitions a
    buffer exactly like separate in-flight batches would."""
    n_per_launch, n_launches = 256, 16
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value
    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == rt.OK, err.value

    data = np.arange(n_per_launch * n_launches, dtype=np.float32)
    buffer = ctypes.c_void_p()
    err = ctypes.c_char_p()
    status = lib.mr_wrap_buffer(
        data.ctypes.data_as(ctypes.c_void_p),
        data.nbytes,
        ctypes.byref(buffer),
        ctypes.byref(err),
    )
    assert status == rt.OK, err.value

    buffers_arr = (ctypes.c_void_p * 1)(buffer)
    offsets_arr = (ctypes.c_size_t * 1)(0)
    desc = rt.LaunchDesc()
    desc.pipeline = pipeline
    desc.buffers = buffers_arr
    desc.buffer_offsets = offsets_arr
    desc.buffer_count = 1
    desc.scalar_count = 0
    desc.threadgroup_memory_count = 0
    desc.grid_x, desc.grid_y, desc.grid_z = n_per_launch, 1, 1
    desc.threadgroup_x, desc.threadgroup_y, desc.threadgroup_z = 0, 0, 0

    batch = ctypes.c_void_p()
    err = ctypes.c_char_p()
    assert lib.mr_batch_create(ctypes.byref(batch), ctypes.byref(err)) == rt.OK, err.value
    for launch_index in range(n_launches):
        offsets_arr[0] = launch_index * n_per_launch * 4
        err = ctypes.c_char_p()
        assert lib.mr_batch_add(batch, ctypes.byref(desc), ctypes.byref(err)) == rt.OK, err.value
    err = ctypes.c_char_p()
    assert lib.mr_batch_commit(batch, ctypes.byref(err)) == rt.OK, err.value
    err = ctypes.c_char_p()
    assert lib.mr_batch_wait(batch, ctypes.byref(err)) == rt.OK, err.value
    lib.mr_release_batch(batch)

    err = ctypes.c_char_p()
    assert lib.mr_buffer_flush_to(buffer, ctypes.byref(err)) == rt.OK, err.value
    np.testing.assert_array_equal(
        data, np.arange(n_per_launch * n_launches, dtype=np.float32) + 1.0
    )
    lib.mr_release_buffer(buffer)

    err = ctypes.c_char_p()
    assert lib.mr_batch_add(None, ctypes.byref(desc), ctypes.byref(err)) != rt.OK
    assert err.value is not None
    assert b"batch is null" in err.value
    lib.mr_free_error_message(err)


_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_MAP_FIXED = 0x10  # <sys/mman.h> on Darwin; the mmap module does not export it


def _map(size: int, at: int | None = None) -> int:
    flags = mmap.MAP_PRIVATE | mmap.MAP_ANON | (_MAP_FIXED if at else 0)
    address = _libc.mmap(at, size, mmap.PROT_READ | mmap.PROT_WRITE, flags, -1, 0)
    assert address not in (None, ctypes.c_void_p(-1).value), os.strerror(ctypes.get_errno())
    return address


def _add_one_in_place(lib, pipeline, address: int, n: int) -> None:
    """Wraps n floats at `address`, adds one on the GPU, releases the wrap."""
    err = ctypes.c_char_p()
    buffer = ctypes.c_void_p()
    status = lib.mr_wrap_buffer(address, n * 4, ctypes.byref(buffer), ctypes.byref(err))
    assert status == rt.OK, err.value
    desc = rt.LaunchDesc()
    desc.pipeline = pipeline
    desc.buffers = (ctypes.c_void_p * 1)(buffer)
    desc.buffer_offsets = (ctypes.c_size_t * 1)(0)
    desc.buffer_count = 1
    desc.grid_x, desc.grid_y, desc.grid_z = n, 1, 1
    desc.threadgroup_x, desc.threadgroup_y, desc.threadgroup_z = 256, 1, 1
    status = lib.mr_dispatch(ctypes.byref(desc), ctypes.byref(err))
    assert status == rt.OK, err.value
    lib.mr_release_buffer(buffer)


@pytest.fixture
def add_one(lib):
    status, library, err = _compile(lib, _ADD_ONE_SOURCE)
    assert status == rt.OK, err.value
    status, pipeline, err = _get_pipeline(lib, library, b"add_one")
    assert status == rt.OK, err.value
    yield pipeline
    lib.mr_release_pipeline(pipeline)
    lib.mr_release_library(library)


def test_wraps_spanning_several_vm_entries_stay_zero_copy(lib, add_one):
    """XNU splits anonymous mappings into entries of at most 128 MB. A wrap
    over 192 MB must alias the memory, not copy it: the GPU's writes are
    visible without mr_buffer_flush_to."""
    size = 192 << 20
    address = _map(size)
    try:
        array = np.ctypeslib.as_array((ctypes.c_float * (size // 4)).from_address(address))
        array[:] = 1.0
        _add_one_in_place(lib, add_one, address, size // 4)
        assert array[0] == array[-1] == 2.0
    finally:
        _libc.munmap(address, size)


def test_cached_wrap_is_not_reused_after_its_range_is_remapped(lib, add_one):
    """A cached wrap keeps the pages it was made over. Once the range is
    unmapped and mapped again at the same address, reusing it would send the
    GPU to the old pages, so the new memory would never change."""
    size = 4 << 20
    address = _map(size)
    view = (ctypes.c_float * (size // 4)).from_address
    np.ctypeslib.as_array(view(address))[:] = 1.0
    _add_one_in_place(lib, add_one, address, size // 4)  # now cached
    _libc.munmap(address, size)
    assert _map(size, at=address) == address
    try:
        array = np.ctypeslib.as_array(view(address))
        array[:] = 10.0
        _add_one_in_place(lib, add_one, address, size // 4)
        assert array[0] == array[-1] == 11.0
        _add_one_in_place(lib, add_one, address, size // 4)  # the fresh wrap is reused
        assert array[0] == array[-1] == 12.0
    finally:
        lib.mr_clear_wrap_cache()
        _libc.munmap(address, size)
