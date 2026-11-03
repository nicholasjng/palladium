"""Quantify SBI's Metal/CPU numerical differences without changing its tolerances.

    JAX_PLATFORMS=metal,cpu uv run python examples/pjrt_sbi_audit.py --output audit.json

Compares ten chained updates and ten updates starting from identical CPU state.
Also isolates simulation/features, gradients on common data, and AdamW on
common gradients. Values are compared in float64 on the host; device work
retains the workload's float32 arithmetic. Nonfinite results fail the audit.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from pjrt_m1_benchmark import _revision, _write_json
from training_steps import adamw, sbi_loss, sbi_trajectory, workload


def errors(actual, reference, rtol: float, atol: float) -> dict:
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} versus {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite audit result")
    absolute = np.abs(a - b)
    excess = np.maximum(absolute - rtol * np.abs(b), 0)
    relative = np.divide(absolute, np.abs(b), out=np.zeros_like(absolute), where=b != 0)
    worst = int(np.argmax(excess)) if excess.size else 0
    return {
        "elements": int(a.size),
        "relative_failures": int(np.count_nonzero(excess > 0)),
        "mixed_failures": int(np.count_nonzero(excess > atol)),
        "zero_reference_mismatches": int(np.count_nonzero((b == 0) & (absolute != 0))),
        "max_absolute_error": float(absolute.max(initial=0)),
        "max_relative_error_nonzero": float(relative.max(initial=0)),
        "required_atol": float(excess.max(initial=0)),
        "worst": {
            "index": [int(i) for i in np.unravel_index(worst, a.shape)] if a.size else [],
            "actual": float(a.flat[worst]) if a.size else None,
            "reference": float(b.flat[worst]) if b.size else None,
        },
    }


def compare(actual, reference, rtol: float, atol: float, step: int) -> list[dict]:
    a, tree_a = jax.tree_util.tree_flatten_with_path(actual)
    b, tree_b = jax.tree_util.tree_flatten_with_path(reference)
    if tree_a != tree_b:
        raise ValueError("audit trees differ")
    return [
        {"path": jax.tree_util.keystr(path), "step": step, **errors(x, y, rtol, atol)}
        for (path, x), (_, y) in zip(a, b, strict=True)
    ]


def summarize(records: list[dict]) -> dict:
    worst = max(records, key=lambda r: r["required_atol"])
    totals = {
        key: sum(r[key] for r in records)
        for key in ("elements", "relative_failures", "mixed_failures", "zero_reference_mismatches")
    }
    maxima = {
        key: max(r[key] for r in records)
        for key in ("max_absolute_error", "max_relative_error_nonzero", "required_atol")
    }
    return {
        **totals,
        **maxima,
        "worst": {"path": worst["path"], "step": worst["step"], **worst["worst"]},
    }


def features(key):
    k1, k2 = jax.random.split(key)
    theta = jax.random.uniform(k1, (512, 4), minval=0.5, maxval=1.5)
    trajectory = sbi_trajectory(theta)
    x = jnp.concatenate([trajectory.mean(0), trajectory.std(0), trajectory[-1]], -1)
    x = x + 0.05 * jax.random.normal(k2, (512, 6))
    return {"theta": theta, "trajectory": trajectory, "features": x}


def audit(count: int = 10, seed: int = 0, rtol: float = 1e-4, atol: float = 2e-6) -> dict:
    cpu, metal = jax.devices("cpu")[0], jax.devices("metal")[0]
    with jax.default_device(cpu):
        fn, params, opt, key = workload("sbi", jax.random.key(seed))
        keys = [jax.random.fold_in(key, i) for i in range(count)]
    report: dict = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "revision": _revision(),
        "device": metal.device_kind,
        "os": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "seed": seed,
        "steps": count,
        "rtol": rtol,
        "atol": atol,
        "required_atol_definition": "max(0, abs(metal-cpu) - rtol*abs(cpu))",
        "modes": {},
        "diagnostics": {},
    }
    state_cpu = jax.device_put((params, opt), cpu)
    state_metal = jax.device_put(state_cpu, metal)
    run = jax.jit(fn)
    records = {"chained": [], "cpu_anchored": []}

    def named(output):
        (p, o), loss = output
        return {"parameters": p, "optimizer": o, "loss": loss}

    for step, key in enumerate(keys, 1):
        reference = run(*state_cpu, jax.device_put(key, cpu))
        chained = run(*state_metal, jax.device_put(key, metal))
        anchored = run(*jax.device_put(state_cpu, metal), jax.device_put(key, metal))
        for name, actual in (("chained", chained), ("cpu_anchored", anchored)):
            records[name].extend(compare(named(actual), named(reference), rtol, atol, step))
        state_cpu, state_metal = reference[0], chained[0]
    for name, rows in records.items():
        report["modes"][name] = {
            "total": summarize(rows),
            "by_step": [
                summarize([r for r in rows if r["step"] == s]) for s in range(1, count + 1)
            ],
            "by_leaf": {
                path: summarize([r for r in rows if r["path"] == path])
                for path in sorted({r["path"] for r in rows})
            },
        }

    def probe(name, function, args):
        compiled = jax.jit(function)
        reference = compiled(*jax.device_put(args, cpu))
        actual = compiled(*jax.device_put(args, metal))
        report["diagnostics"][name] = compare(actual, reference, rtol, atol, 1)
        return reference

    data = probe("simulation_and_features", features, (keys[0],))
    loss, grads = probe(
        "gradients_on_common_data",
        jax.value_and_grad(sbi_loss),
        (params, data["features"], data["theta"]),
    )
    probe("adamw_on_common_gradients", adamw, (params, grads, opt))
    report["diagnostic_cpu_loss"] = float(loss)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    report = audit(args.steps, args.seed)
    if args.output:
        _write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
