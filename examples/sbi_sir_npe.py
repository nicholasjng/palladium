"""Example 8: neural posterior estimation on GPU-simulated SIR data.

Docs: examples/sbi_sir.py for the simulator.
Simulation-based inference is simulate, train a neural posterior
estimator on the (theta, x) pairs, then sample it at the observation.
The GPU touches only the first stage. This example measures two things:

1. Interchangeability of GPU- and CPU-simulated data. sbi's NPE is
   trained on the same draws from three simulators: Palladium, the same
   RK4 scheme vectorized in NumPy (the strong CPU baseline), and a
   per-draw SciPy solver (the pattern sbi's own tutorials use). Each
   posterior is sampled at one observation and compared with Palladium's
   by the classifier two-sample test (C2ST; 0.5 means indistinguishable).
2. The split of wall clock between simulation and training for the
   full-size Palladium dataset.

A posterior predictive check pushes posterior draws back through the
Palladium simulator and compares predicted infected counts with the
observation.

NPE training runs on CPU torch, so torch-MPS normalizing flows are not
a variable. Run with

    uv run --with sbi examples/sbi_sir_npe.py [--n 20000] [--compare-n 2000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbi_sir import (
    PRIOR_LOC,
    PRIOR_SCALE,
    as_sbi_simulator,
    numpy_simulator,
    observe,
    palladium_simulator,
    scipy_simulator,
)

SIMULATORS = (
    ("palladium", palladium_simulator),
    ("numpy", numpy_simulator),
    ("scipy", scipy_simulator),
)


def make_prior():
    import torch
    from sbi.utils import process_prior

    prior, *_ = process_prior(
        torch.distributions.Independent(
            torch.distributions.LogNormal(torch.as_tensor(PRIOR_LOC), torch.as_tensor(PRIOR_SCALE)),
            1,
        )
    )
    return prior


def simulate(prior, fractions, n, seed):
    from sbi.inference import simulate_for_sbi

    simulator = as_sbi_simulator(fractions)
    start = time.perf_counter()
    theta, x = simulate_for_sbi(
        simulator, prior, n, seed=seed, simulation_batch_size=None, show_progress_bar=False
    )
    return theta, x, time.perf_counter() - start


def train_npe(prior, theta, x, *, seed, max_epochs):
    import torch
    from sbi.inference import NPE

    torch.manual_seed(seed)
    inference = NPE(prior, show_progress_bars=False)
    start = time.perf_counter()
    estimator = inference.append_simulations(theta, x).train(max_num_epochs=max_epochs)
    seconds = time.perf_counter() - start
    return inference.build_posterior(estimator), seconds


def observation(theta_true, seed=101):
    fractions = palladium_simulator(np.asarray([theta_true], np.float32))
    return observe(fractions, seed=seed)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=20_000, help="Palladium draws for the main NPE")
    parser.add_argument("--compare-n", type=int, default=2_000, help="draws per simulator for C2ST")
    parser.add_argument("--posterior-samples", type=int, default=5_000)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plot", default=None, help="write the predictive check to this PNG")
    args = parser.parse_args()

    import torch
    from sbi.utils.metrics import c2st

    prior = make_prior()
    theta_true = np.array([0.4, 0.125], np.float32)  # the prior medians
    x_o = torch.as_tensor(observation(theta_true))
    print(f"observation from theta={theta_true.tolist()}: {x_o.to(torch.int64).tolist()}")

    # 1. Interchangeability: same draws, three simulators, one observation.
    # The observation model quantizes I/N into counts of 1000, so simulators
    # that agree to a few 1e-6 in the fraction usually produce bit-identical
    # datasets and identical estimators. C2ST is reported only when the
    # datasets differ; the metric misbehaves on identical sample sets.
    print(f"\ninterchangeability at {args.compare_n} draws per simulator")
    datasets, posteriors = {}, {}
    for name, fractions in SIMULATORS:
        datasets[name] = simulate(prior, fractions, args.compare_n, args.seed)
    theta_p, x_p, _ = datasets["palladium"]
    fractions_p = palladium_simulator(theta_p.numpy())
    for name, fractions in SIMULATORS[1:]:
        theta, x, _ = datasets[name]
        assert torch.equal(theta_p, theta), "seeded proposals must match"
        gap = np.max(np.abs(fractions_p - fractions(theta.numpy())))
        differing = int((x_p != x).sum())
        print(
            f"  palladium vs {name:6s} max |I/N| deviation {gap:.2e}; "
            f"{differing} of {x_p.numel()} observed counts differ"
        )
    for name, _ in SIMULATORS:
        theta, x, t_sim = datasets[name]
        posterior, t_train = train_npe(prior, theta, x, seed=args.seed, max_epochs=args.max_epochs)
        samples = posterior.sample((args.posterior_samples,), x=x_o, show_progress_bars=False)
        posteriors[name] = samples
        mean, std = samples.mean(0).tolist(), samples.std(0).tolist()
        print(
            f"  {name:10s} simulate {t_sim:7.3f} s  train {t_train:7.1f} s  "
            f"posterior mean {mean[0]:.4f} {mean[1]:.4f}  std {std[0]:.4f} {std[1]:.4f}"
        )
    for name, _ in SIMULATORS[1:]:
        if torch.equal(x_p, datasets[name][1]):
            print(
                f"  palladium and {name} datasets are identical, posteriors identical by construction"
            )
        else:
            score = float(c2st(posteriors["palladium"], posteriors[name]))
            print(f"  C2ST(palladium, {name}) = {score:.3f}  (0.5 = indistinguishable)")

    # 2. Where the time goes at full size.
    print(f"\nfull run, {args.n} Palladium draws")
    theta, x, t_sim = simulate(prior, palladium_simulator, args.n, args.seed)
    _, _, t_numpy = simulate(prior, numpy_simulator, args.n, args.seed)
    posterior, t_train = train_npe(prior, theta, x, seed=args.seed, max_epochs=args.max_epochs)
    samples = posterior.sample((args.posterior_samples,), x=x_o, show_progress_bars=False)
    lo, hi = np.quantile(samples.numpy(), [0.025, 0.975], axis=0)
    covered = bool(np.all((lo <= theta_true) & (theta_true <= hi)))
    print(
        f"  simulate {t_sim:.3f} s (numpy {t_numpy:.3f} s), train {t_train:.1f} s, "
        f"sample {args.posterior_samples}"
    )
    print(
        f"  posterior mean {samples.mean(0).tolist()}  95% interval {lo.tolist()} .. {hi.tolist()}"
    )
    print(f"  true theta inside the 95% interval: {covered}")

    # 3. Posterior predictive check through the GPU simulator.
    predicted = observe(palladium_simulator(samples.numpy()), seed=args.seed + 1)
    q05, q50, q95 = np.quantile(predicted, [0.05, 0.5, 0.95], axis=0)
    inside = np.mean((q05 <= x_o.numpy()) & (x_o.numpy() <= q95))
    print(f"\nposterior predictive check: {inside:.0%} of observed counts inside the 5%..95% band")
    print("  t(days)  observed  predicted median [5%, 95%]")
    for k in range(len(q50)):
        print(f"  {17 * k:5d}   {int(x_o[k]):7d}   {int(q50[k]):7d} [{int(q05[k])}, {int(q95[k])}]")
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = 17 * np.arange(len(q50))
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.fill_between(days, q05, q95, alpha=0.3, label="posterior predictive 5%..95%")
        ax.plot(days, q50, label="predictive median")
        ax.plot(days, x_o.numpy(), "o", label="observation")
        ax.set_xlabel("day")
        ax.set_ylabel("infected of 1000 sampled")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"  wrote {args.plot}")


if __name__ == "__main__":
    main()
