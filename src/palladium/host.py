"""Work the `metal` plugin runs on the host: Python callbacks and LAPACK.

A lowering rule here emits a `palladium_host` custom call that names a Python
function by index. The plugin's compiler turns the call into a host step: the
plugin waits for the GPU to compute the inputs, runs the function on arrays
that share the buffers' unified memory, and goes on with the results.

`jax.pure_callback`, `jax.experimental.io_callback`, `jax.debug.callback`,
`jax.debug.print` and `checkify` checks lower this way, as does dense linear
algebra (Cholesky, triangular solves, LU, QR and its Householder form,
symmetric eigendecomposition, SVD, and the Hessenberg, tridiagonal and real
Schur forms), which runs on LAPACK through NumPy and SciPy. A host step ends the
command buffer in flight, so a program with one does not pipeline.
"""

from __future__ import annotations

import dataclasses
import itertools
import weakref
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from jax._src import (  # the primitives and their rules have no public names
    callback as jax_callback,
    checkify,
    debugging,
    effects,
    ffi,
)
from jax._src.interpreters import mlir
from jax._src.lax import linalg

TARGET = "palladium_host"


@dataclasses.dataclass(eq=False)
class _Function:
    name: str
    function: Callable[..., Sequence[np.ndarray]]


# Functions by the index their custom call carries. The lowered module keeps
# each alive until it is compiled, when the program takes a reference.
_functions: weakref.WeakValueDictionary[int, _Function] = weakref.WeakValueDictionary()
_indices = itertools.count()


def registered(index: int) -> tuple[str, Callable[..., Sequence[np.ndarray]]]:
    """The name and function a `palladium_host` call with `index` runs."""
    entry = _functions[index]
    return entry.name, entry.function


def emit(
    ctx: mlir.LoweringRuleContext,
    name: str,
    function: Callable[..., Sequence[np.ndarray]],
    operands: Sequence[Any],
    *,
    effect: bool = False,
) -> Sequence[Any]:
    """Emit a call of `function` on the host and return its MLIR results.

    Parameters
    ----------
    ctx : mlir.LoweringRuleContext
        The rule's context; `avals_out` gives the result types.
    name : str
        What error messages call the function.
    function : callable
        Takes the operands as NumPy arrays and returns one array per result,
        of the result's dtype and shape.
    operands : sequence
        The MLIR operands.
    effect : bool
        The call runs even if nothing reads its results.
    """
    entry = _Function(name, function)
    index = next(_indices)
    _functions[index] = entry
    ctx.module_context.add_keepalive(entry)
    rule = ffi.build_ffi_lowering_function(TARGET, has_side_effect=effect)
    return rule(ctx, *operands, index=np.uint64(index)).results


def _typed(function: Callable, ctx: mlir.LoweringRuleContext) -> Callable:
    """`function` with its results as arrays of the rule's output dtypes."""
    dtypes = [np.dtype(aval.dtype) for aval in ctx.avals_out]

    def typed(*args):
        results = function(*args)
        return [np.asarray(r).astype(d, copy=False) for r, d in zip(results, dtypes, strict=True)]

    return typed


def _keep_tokens(ctx: mlir.LoweringRuleContext, effect) -> None:
    """Pass an ordered effect's token through: host steps run in program order."""
    if effects.ordered_effects.contains(effect):
        ctx.set_tokens_out(ctx.tokens_in)


def _pure_callback(ctx, *args, callback, sharding, **params):
    def function(*arrays):
        return jax_callback.pure_callback_impl(*arrays, callback=callback, sharding=None, **params)

    return emit(ctx, "jax.pure_callback", _typed(function, ctx), args)


def _io_callback(ctx, *args, callback, sharding, ordered, **params):
    def function(*arrays):
        return jax_callback.io_callback_impl(
            *arrays, callback=callback, sharding=None, ordered=ordered, **params
        )

    if ordered:
        ctx.set_tokens_out(ctx.tokens_in)
    return emit(ctx, "jax.experimental.io_callback", _typed(function, ctx), args, effect=True)


def _debug_callback(ctx, *args, effect, partitioned, callback, **params):
    def function(*arrays):
        debugging.debug_callback_p.impl(
            *arrays, effect=effect, partitioned=partitioned, callback=callback, **params
        )
        return ()

    _keep_tokens(ctx, effect)
    return emit(ctx, "jax.debug.callback", function, args, effect=True)


def _debug_print(
    ctx,
    *args,
    fmt,
    ordered,
    partitioned,
    in_tree,
    static_args,
    np_printoptions,
    has_placeholders,
    logging_record,
):
    import functools

    callback = functools.partial(
        debugging._format_print_callback,
        fmt,
        dict(np_printoptions),
        has_placeholders,
        logging_record,
    )
    callback = debugging._make_flat_callback(in_tree, callback, static_args)
    effect = debugging.ordered_debug_effect if ordered else debugging.debug_effect
    return _debug_callback(ctx, *args, effect=effect, partitioned=partitioned, callback=callback)


def _check(ctx, *args, err_tree, debug):
    if debug:
        return []

    def function(*arrays):
        checkify.python_err(err_tree, *arrays)
        return ()

    return emit(ctx, "checkify.check", function, args, effect=True)


def _batched(function: Callable, cores: Sequence[int], *arrays: np.ndarray) -> list[np.ndarray]:
    """Apply `function` to each batch element of `arrays`, whose trailing
    `cores` dimensions are the operand, and stack each of its results."""
    batch = arrays[0].shape[: arrays[0].ndim - cores[0]]
    count = int(np.prod(batch, dtype=np.int64))
    flat = [a.reshape(count, *a.shape[a.ndim - c :]) for a, c in zip(arrays, cores, strict=True)]
    results = [function(*(a[k] for a in flat)) for k in range(count)]
    return [np.stack(r).reshape(*batch, *r[0].shape) for r in zip(*results, strict=True)]


def _cholesky(ctx, x):
    def function(a):
        # Only the lower triangle is read, and a matrix that is not positive
        # definite gives NaNs, as on other backends.
        a = np.tril(a) + np.swapaxes(np.tril(a, -1), -1, -2)
        try:
            return [np.linalg.cholesky(a)]
        except np.linalg.LinAlgError:
            pass

        def one(m):
            try:
                return [np.linalg.cholesky(m)]
            except np.linalg.LinAlgError:
                return [np.full_like(m, np.nan)]

        return _batched(one, (2,), a)

    return emit(ctx, "cholesky", _typed(function, ctx), [x])


def _substitute(a: np.ndarray, b: np.ndarray, lower: bool, unit: bool) -> np.ndarray:
    """Solve a x = b for triangular `a`, a row at a time over all batches."""
    n = a.shape[-1]
    x = np.zeros_like(b)
    for i in range(n) if lower else range(n - 1, -1, -1):
        done = slice(0, i) if lower else slice(i + 1, n)
        row = b[..., i, :] - np.einsum("...k,...kj->...j", a[..., i, done], x[..., done, :])
        x[..., i, :] = row if unit else row / a[..., i, i, None]
    return x


def _triangular_solve(ctx, a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
    def function(a, b):
        if not left_side:
            # x a = b is a' x' = b'.
            a, b = np.swapaxes(a, -1, -2), np.swapaxes(b, -1, -2)
        low = lower != (transpose_a != (not left_side))
        if transpose_a:
            a = np.swapaxes(a, -1, -2)
        # Under vmap, one operand may lack the other's batch dimensions.
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a, b = np.broadcast_to(a, batch + a.shape[-2:]), np.broadcast_to(b, batch + b.shape[-2:])
        with np.errstate(divide="ignore", invalid="ignore"):
            x = _substitute(a, b, low, unit_diagonal)
        return [x if left_side else np.swapaxes(x, -1, -2)]

    return emit(ctx, "triangular_solve", _typed(function, ctx), [a, b])


def _lu(ctx, x):
    import scipy.linalg

    def one(a):
        (getrf,) = scipy.linalg.get_lapack_funcs(("getrf",), (a,))
        lu, pivots, _ = getrf(a)
        permutation = np.arange(a.shape[0], dtype=np.int32)
        for i, p in enumerate(pivots):
            permutation[[i, p]] = permutation[[p, i]]
        return lu, pivots.astype(np.int32), permutation

    return emit(ctx, "lu", _typed(lambda a: _batched(one, (2,), a), ctx), [x])


def _qr(ctx, x, *, pivoting, full_matrices, use_magma):
    import scipy.linalg

    mode = "full" if full_matrices else "economic"

    def one(a):
        if not pivoting:
            return scipy.linalg.qr(a, mode=mode, check_finite=False)
        return scipy.linalg.qr(a, mode=mode, pivoting=True, check_finite=False)

    return emit(ctx, "qr", _typed(lambda a: _batched(one, (2,), a), ctx), [x])


def _eigh(ctx, x, *, lower, sort_eigenvalues, subset_by_index, algorithm=None):
    n = ctx.avals_in[0].shape[-1]

    def function(a):
        w, v = np.linalg.eigh(a, UPLO="L" if lower else "U")
        if subset_by_index is not None and tuple(subset_by_index) != (0, n):
            start, end = subset_by_index
            w, v = w[..., start:end], v[..., start:end]
        return v, w

    return emit(ctx, "eigh", _typed(function, ctx), [x])


def _svd(ctx, x, *, full_matrices, compute_uv, subset_by_index, algorithm=None):
    def function(a):
        if not compute_uv:
            s = np.linalg.svd(a, compute_uv=False)
            return [s if subset_by_index is None else s[..., slice(*subset_by_index)]]
        u, s, vt = np.linalg.svd(a, full_matrices=full_matrices)
        if subset_by_index is not None:
            keep = slice(*subset_by_index)
            u, s, vt = u[..., keep], s[..., keep], vt[..., keep, :]
        return s, u, vt

    return emit(ctx, "svd", _typed(function, ctx), [x])


def _lapack(ctx, name: str, operands, routine: str, results, **options):
    """A host call of one LAPACK routine per batch element. `results` picks,
    from the routine's return values, the ones the primitive returns."""
    import scipy.linalg

    cores = (2, 1)[: len(operands)]

    def one(*arrays):
        (function,) = scipy.linalg.get_lapack_funcs((routine,), arrays[:1])
        returned = function(*arrays, **options)
        return [returned[k] for k in results]

    return emit(ctx, name, _typed(lambda *arrays: _batched(one, cores, *arrays), ctx), operands)


def _geqrf(ctx, a):
    return _lapack(ctx, "geqrf", [a], "geqrf", (0, 1))


def _householder_product(ctx, a, taus):
    return _lapack(ctx, "householder_product", [a, taus], "orgqr", (0,))


def _hessenberg(ctx, a):
    return _lapack(ctx, "hessenberg", [a], "gehrd", (0, 1))


def _tridiagonal(ctx, a, *, lower):
    return _lapack(ctx, "tridiagonal", [a], "sytrd", (0, 1, 2, 3), lower=lower)


def _schur(ctx, x, *, compute_schur_vectors, sort_eig_vals, select_callable):
    import scipy.linalg

    if sort_eig_vals:
        raise NotImplementedError("schur with sorted eigenvalues is not implemented on metal")

    def one(a):
        form, vectors = scipy.linalg.schur(a, output="real", check_finite=False)
        return (form, vectors) if compute_schur_vectors else (form,)

    return emit(ctx, "schur", _typed(lambda a: _batched(one, (2,), a), ctx), [x])


def register() -> None:
    """Register the host lowering rules for the `metal` platform."""
    rules = {
        jax_callback.pure_callback_p: _pure_callback,
        jax_callback.io_callback_p: _io_callback,
        debugging.debug_callback_p: _debug_callback,
        debugging.debug_print_p: _debug_print,
        checkify.check_p: _check,
        linalg.cholesky_p: _cholesky,
        linalg.triangular_solve_p: _triangular_solve,
        linalg.lu_p: _lu,
        linalg.qr_p: _qr,
        linalg.eigh_p: _eigh,
        linalg.svd_p: _svd,
        linalg.geqrf_p: _geqrf,
        linalg.householder_product_p: _householder_product,
        linalg.hessenberg_p: _hessenberg,
        linalg.tridiagonal_p: _tridiagonal,
        linalg.schur_p: _schur,
    }
    for primitive, rule in rules.items():
        # A rule that closes over a Python function must run for every call.
        mlir.register_lowering(primitive, rule, platform="metal", cacheable=False)
