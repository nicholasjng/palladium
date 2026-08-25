"""Differential verification against a reference implementation.

The project's validation doctrine is "diff every kernel against the
`interpret=True` oracle with f32-honest tolerances", and until now every
consumer hand-rolled it -- 40-odd copies of the same
`assert_allclose(f(x), f.interpret(x))` across the test suite. This is
that comparison as one call, with the two facts that make it correct
built in rather than left to each caller.

Fact one: FAST math is the default, and it reorders float arithmetic and
uses approximate transcendentals, so results are not bit-equal to the
CPU oracle. The default tolerance reflects that.

Fact two: cooperative kernels have no interpret oracle at all. Interpret
runs program instances sequentially with no notion of a threadgroup, so
it reports `thread_index() == 0` and `threads_per_threadgroup() == 1` --
a kernel that reduces across threads computes something *different*
there. Comparing against it would be meaningless, so `verify` refuses
and asks for an explicit `reference=` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["VerificationError", "verify_against"]

# FAST math reorders arithmetic and approximates transcendentals; this is
# the tolerance the existing suite settled on for f32 kernels after
# measuring, not a guess.
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-6


class VerificationError(AssertionError):
    """A kernel's GPU output disagreed with its reference.

    Subclasses AssertionError so pytest reports it as a plain assertion
    failure, while still being catchable as a distinct type.

    Attributes
    ----------
    output_index : int
        Which output disagreed, for multi-output kernels.
    mismatches, size : int
        Count of elements outside tolerance, and total element count.
    worst_index : tuple of int or None
        Index of the largest deviation.
    got, want : float or None
        The two values at `worst_index`.
    """

    def __init__(
        self,
        message: str,
        *,
        output_index: int = 0,
        mismatches: int = 0,
        size: int = 0,
        worst_index: tuple[int, ...] | None = None,
        got: float | None = None,
        want: float | None = None,
    ) -> None:
        super().__init__(message)
        self.output_index = output_index
        self.mismatches = mismatches
        self.size = size
        self.worst_index = worst_index
        self.got = got
        self.want = want


def _as_tuple(outs: Any) -> tuple[np.ndarray, ...]:
    if isinstance(outs, (tuple, list)):
        return tuple(np.asarray(o) for o in outs)
    return (np.asarray(outs),)


def _report(
    got: np.ndarray, want: np.ndarray, index: int, rtol: float, atol: float
) -> None:
    if got.shape != want.shape:
        raise VerificationError(
            f"output {index}: shape {got.shape} from the GPU, {want.shape} "
            "from the reference",
            output_index=index,
        )
    close = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=True)
    if close.all():
        return

    diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
    diff[np.isnan(diff)] = np.inf
    flat = int(np.argmax(diff))
    worst = np.unravel_index(flat, got.shape)
    n_bad, size = int((~close).sum()), got.size
    g, w = got[worst], want[worst]
    scale = abs(float(w))
    rel = f"{abs(float(g) - float(w)) / scale:.3g}" if scale else "n/a"
    raise VerificationError(
        f"output {index}: {n_bad} of {size} elements outside "
        f"rtol={rtol:g}/atol={atol:g}.\n"
        f"  worst at {tuple(int(i) for i in worst)}: "
        f"GPU {float(g)!r} vs reference {float(w)!r} (relative {rel})\n"
        "  If the kernel uses transcendentals, FAST math (the default) "
        "approximates them; try math_mode=MathMode.SAFE to tell a real "
        "disagreement from float slop.",
        output_index=index,
        mismatches=n_bad,
        size=size,
        worst_index=tuple(int(i) for i in worst),
        got=float(g),
        want=float(w),
    )


def verify_against(
    gpu_fn: Callable[..., Any],
    oracle: Callable[..., Any] | None,
    uses_threadgroup: bool,
    args: tuple,
    reference: Callable[..., Any] | None,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, ...]:
    """Run `gpu_fn` and a reference over `args`; raise on disagreement.

    Shared by `MetalCallable.verify` and `FfiCallable.verify`; see
    `MetalCallable.verify` for the user-facing contract.
    """
    if reference is None:
        if uses_threadgroup:
            raise VerificationError(
                "this kernel uses palladium.threadgroup_memory, so the "
                "interpret oracle cannot validate it: interpret runs program "
                "instances sequentially with no threadgroup, reporting "
                "thread_index() == 0 and threads_per_threadgroup() == 1, so a "
                "kernel that reduces across threads computes something else "
                "there. Pass reference=<callable> with an implementation that "
                "models the real threadgroup semantics."
            )
        if oracle is None:  # pragma: no cover - both paths supply one
            raise VerificationError("no reference and no interpret oracle")
        reference = oracle

    got = _as_tuple(gpu_fn(*args))
    want = _as_tuple(reference(*args))
    if len(got) != len(want):
        raise VerificationError(
            f"kernel returned {len(got)} outputs, reference returned {len(want)}"
        )
    for i, (g, w) in enumerate(zip(got, want, strict=True)):
        _report(g, w, i, rtol, atol)
    return got
