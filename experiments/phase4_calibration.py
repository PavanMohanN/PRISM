"""PRISM Phase 4 -- E8: calibration breadth (guide item 17).

The strongest calibration evidence was concentrated on the three SBI tasks. Here
we run simulation-based calibration (SBC) + coverage on ALL domains, adding
Oscillator and Lotka-Volterra (dynamical systems); because the data are
procedurally generated, repeated truth draws are free. We record a calibration
p-value, coverage, ECE, and (where a reference exists) C2ST, for PRISM and the
two learned baselines, plus the SBC ranks / reliability curve for the figure.

Run in D:\\ICLR:  py experiments\\phase4_calibration.py --mode smoke
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from experiments.provenance import ResultManifest
from experiments._common import load_cfg
from experiments.api import make_method
from experiments.api import get_dataset, emcee_reference
from experiments.phase4_analysis import (
    sbc_ranks, sbc_uniformity, coverage_curve, ece_from_curve,
)
from prism.posterior.calibration import c2st, gaussian_nll

TASKS = ["gaussian_linear", "slcp", "two_moons", "oscillator", "lotka_volterra"]
METHODS = ["PRISM", "NPE-CNF", "VI"]
SEEDS = [0, 1, 2, 3, 4]
N_SBC = 128          # observations for SBC (raise in --full)
L = 200              # posterior samples per observation


def _reference(task, y0, n):
    if task in ("slcp", "gaussian_linear"):
        return emcee_reference(task, y0)
    return None      # oscillator / lotka / two_moons assessed by SBC+coverage


def run(task, method, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    est = make_method(method, task, cfg)
    est.fit(ds.X_train, ds.y_train)

    n = min(N_SBC, len(ds.y_test))
    truths = np.asarray(ds.X_test[:n])
    post = np.stack([np.asarray(est.predict_posterior(ds.y_test[i], L)).reshape(-1, dim)[:L]
                     for i in range(n)], 0)                 # [n, L, d]

    ranks = sbc_ranks(truths, post)
    _, sbc_p = sbc_uniformity(ranks)
    levels, emp = coverage_curve(truths, post, (0.5, 0.8, 0.9, 0.95))
    ece = ece_from_curve(levels, emp)
    cov90 = emp[levels.index(0.9)] if 0.9 in levels else emp[-2]

    out = {"sbc_p": sbc_p, "cov90": float(cov90), "ece": float(ece),
           "nll_phys": float(gaussian_nll(post[0], truths[0])),
           "sbc_ranks": ranks.ravel().astype(float).tolist(),
           "rel_levels": list(levels), "rel_emp": list(map(float, emp))}
    ref = _reference(task, ds.y_test[0], 1000)
    if ref is not None:
        out["c2st"] = float(c2st(post[0], ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--manifest", default="results/manifest.json")
    args = ap.parse_args()
    cfg = load_cfg("phase4", args.mode)

    import os
    man = ResultManifest(args.manifest)
    if os.path.exists(args.manifest):
        man.load()
    for task in TASKS:
        for method in METHODS:
            man.run_seeds("phase4_calibration", task, method,
                          fn=lambda s, m=method, t=task: run(t, m, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        p = man.agg("phase4_calibration", task, "PRISM", "sbc_p")[0]
        e = man.agg("phase4_calibration", task, "PRISM", "ece")[0]
        print(f"  [{task:16s}] PRISM  SBC p={p:.2f}  ECE={e:.3f}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
