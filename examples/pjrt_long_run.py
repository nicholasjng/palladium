"""Does long training on `metal` reach the same losses as on CPU? (M2's first target.)

    JAX_PLATFORMS=metal,cpu uv run python examples/pjrt_long_run.py --workload sbi \\
        --steps 5000 --output run.json

Float32 training amplifies rounding differences, so after enough steps no two
runs agree bitwise, even on one backend. The yardstick is CPU against itself:
the workload trains on CPU, on CPU again from initial parameters moved by one
float32 ulp (`--perturbations` times, in different directions), and on Metal,
all from the same data. Metal agrees with CPU if it stays as close to the CPU
run as the perturbed CPU runs do.

At each checkpoint the script compares, against the CPU run: the training
loss averaged over the last `--window` steps, the parameters (relative L2
distance over all leaves), and the loss on a fixed held-out batch, evaluated
on CPU for every run so that only training differs.

Workloads: `sbi`, the SBI step of `training_steps.py` (RK4 simulation, MDN
posterior, AdamW; fresh simulations every step); `cnf`, the CNF update of
`cnf_density.py` (256 points, Adam), trained on Metal with Palladium's adjoint
kernels and on CPU in plain JAX.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import time
from pathlib import Path

import cnf_density
import jax
import jaxlib
import numpy as np
from pjrt_m1_benchmark import _revision, _write_json
from pjrt_sbi_audit import features
from training_steps import sbi_loss, sbi_step, workload


def _sbi(seed: int) -> dict:
    with jax.default_device(jax.devices("cpu")[0]):
        _, params, opt, key = workload("sbi", jax.random.key(seed))
        held_out = features(jax.random.key(seed + 1_000_003))

    def work(state):
        run = jax.jit(lambda state, i: sbi_step(*state, jax.random.fold_in(key, i)))
        evaluate = jax.jit(lambda p: sbi_loss(p, held_out["features"], held_out["theta"]))
        return {
            "state": state,
            "step": lambda device: run,
            "params": lambda state: state[0],
            "evaluate": evaluate,
        }

    params64, (m64, v64) = jax.tree.map(lambda x: np.asarray(x, np.float64), (params, opt[:2]))
    return work((params, opt)) | {"float64": lambda: work((params64, (m64, v64, opt[2])))}


def _cnf(seed: int) -> dict:
    n = 256
    with jax.default_device(jax.devices("cpu")[0]):
        state = cnf_density.initial_state(seed=seed)
        data, held_out = cnf_density.mixture_data(n, seed), cnf_density.mixture_data(n, seed + 12)
        _, loss = cnf_density.make_training_step(n, variant="jax")

    def step(device):
        variant = "jax" if device.platform == "cpu" else "reverse"
        update, _ = cnf_density.make_training_step(n, variant=variant)
        points = jax.device_put(data, device)
        return lambda state, i: update(state, points)

    return {
        "state": state,
        "step": step,
        "params": lambda state: state[0],
        "evaluate": lambda p: loss(p, held_out),
    }


WORKLOADS = {"sbi": _sbi, "cnf": _cnf}


def perturbed(params, direction: int, seed: int):
    """`params` with every element moved by one float32 ulp.

    Direction +1 or -1 moves all elements up or down; 0 picks each element's
    direction at random.
    """
    rng = np.random.default_rng(seed)

    def move(leaf):
        leaf = np.asarray(leaf)
        sign = rng.choice([-1.0, 1.0], leaf.shape) if direction == 0 else float(direction)
        return np.nextafter(leaf, np.asarray(sign * np.inf, leaf.dtype))

    return jax.tree.map(move, params)


def train(work: dict, state, device, steps: int, checkpoints: list[int]) -> dict:
    """Pipelined training: losses and checkpoint parameters are fetched at the end."""
    run = work["step"](device)
    state = jax.device_put(state, device)
    losses, saved = [], {}
    start = time.perf_counter()
    for i in range(steps):
        state, loss = run(state, i)
        losses.append(loss)
        if i + 1 in checkpoints:
            saved[i + 1] = work["params"](state)
    jax.block_until_ready(state)
    seconds = time.perf_counter() - start
    cpu = jax.devices("cpu")[0]
    params = {k: jax.device_put(p, cpu) for k, p in saved.items()}
    return {
        "losses": np.array([float(x) for x in losses]),
        "params": params,
        "held_out": {k: float(work["evaluate"](p)) for k, p in params.items()},
        "seconds": seconds,
    }


def distance(params, reference) -> float:
    """Relative L2 distance over all leaves."""
    a, b = (np.concatenate([np.ravel(x) for x in jax.tree.leaves(p)]) for p in (params, reference))
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def compare(run: dict, reference: dict, step: int, window: int) -> dict:
    """Differences from the reference run at one checkpoint, relative to it."""
    recent = slice(max(0, step - window), step)
    mean, ref_mean = run["losses"][recent].mean(), reference["losses"][recent].mean()
    held, ref_held = run["held_out"][step], reference["held_out"][step]
    return {
        "loss_mean": float(mean),
        "loss_mean_difference": float(abs(mean - ref_mean) / abs(ref_mean)),
        "parameter_distance": distance(run["params"][step], reference["params"][step]),
        "held_out_loss": held,
        "held_out_difference": float(abs(held - ref_held) / abs(ref_held)),
    }


def long_run(
    name: str, steps: int, every: int, window: int, perturbations: int, seed: int, losses: bool
) -> dict:
    work = WORKLOADS[name](seed)
    checkpoints = sorted({*range(every, steps + 1, every), steps})
    cpu, metal = jax.devices("cpu")[0], jax.devices("metal")[0]
    reference = train(work, work["state"], cpu, steps, checkpoints)
    truth = None
    if "float64" in work:
        # The same data in float64 on CPU: which float32 run is more accurate?
        with jax.enable_x64(True):
            exact = work["float64"]()
            truth = train(exact, exact["state"], cpu, steps, checkpoints)
    params, *rest = work["state"]
    runs = {"metal": train(work, work["state"], metal, steps, checkpoints)}
    for k, direction in zip(range(perturbations), (1, -1, 0, 0, 0, 0, 0), strict=False):
        start = (perturbed(params, direction, seed + k), *rest)
        runs[f"cpu_ulp_{k}"] = train(work, start, cpu, steps, checkpoints)
    table = {
        step: {run: compare(result, reference, step, window) for run, result in runs.items()}
        for step in checkpoints
    }
    return {
        "workload": name,
        "steps": steps,
        "window": window,
        "seed": seed,
        "seconds": {"cpu": reference["seconds"]}
        | {run: result["seconds"] for run, result in runs.items()},
        "cpu": {
            step: {
                "loss_mean": float(reference["losses"][max(0, step - window) : step].mean()),
                "held_out_loss": reference["held_out"][step],
            }
            for step in checkpoints
        },
        "checkpoints": table,
        "versus_float64": None
        if truth is None
        else {
            step: {
                run: compare(result, truth, step, window)
                for run, result in {"cpu": reference, **runs}.items()
            }
            for step in checkpoints
        },
    } | (
        {
            "losses": {"cpu": reference["losses"].tolist()}
            | {run: result["losses"].tolist() for run, result in runs.items()}
        }
        if losses
        else {}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="sbi")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--every", type=int, default=500, help="checkpoint interval")
    parser.add_argument("--window", type=int, default=100, help="steps averaged per loss")
    parser.add_argument("--perturbations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--losses", action="store_true", help="also record every step's loss")
    args = parser.parse_args()
    if min(args.steps, args.every, args.window, args.perturbations) < 1:
        parser.error("steps, every, window and perturbations must be positive")
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "device": jax.devices("metal")[0].device_kind,
        "os": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "revision": _revision(),
        **long_run(
            args.workload,
            args.steps,
            args.every,
            args.window,
            args.perturbations,
            args.seed,
            args.losses,
        ),
    }
    print(
        f"{args.workload}: seconds "
        + ", ".join(f"{k} {v:.2f}" for k, v in result["seconds"].items())
    )
    tables = {"cpu": result["checkpoints"], "float64": result["versus_float64"]}
    for versus, table in tables.items():
        if table is None:
            continue
        columns = list(next(iter(table.values())))
        for metric in ("loss_mean_difference", "parameter_distance", "held_out_difference"):
            print(f"\n{metric} versus {versus}")
            print(f"{'step':>6s} " + " ".join(f"{run:>12s}" for run in columns))
            for step, row in table.items():
                print(f"{step:6d} " + " ".join(f"{row[run][metric]:12.3e}" for run in columns))
    if args.output:
        _write_json(args.output, json.loads(json.dumps(result)))


if __name__ == "__main__":
    main()
