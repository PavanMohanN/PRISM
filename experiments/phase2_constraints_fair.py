"""PRISM Phase 2 -- E2: fair constraint comparison (guide item 10).

Zero violations is a property of the support transform, not of PRISM. We give the
SAME transform to cINN, NPE-CNF, and VI, then compare on what actually
differentiates the methods: calibration, physical-space NLL, inverse error, and
latency. After wrapping, all methods should show ~0 violations; the story moves
off "violations" and onto posterior quality.

Run in D:\\ICLR:  py experiments\\phase2_constraints_fair.py --mode smoke
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch

from experiments.provenance import ResultManifest
from experiments._common import load_cfg
from experiments.api import make_method
from experiments.api import get_dataset, emcee_reference
from baselines.support_wrapper import SupportConstrained
from prism.constraints.transforms import make_transform
from prism.posterior.calibration import c2st, coverage, gaussian_nll   # existing repo

# task -> (constraint kind, kwargs)
CONSTRAINED = {
    "oscillator":     ("positive", {}),
    "lotka_volterra": ("positive", {}),
    "two_moons":      ("box", {"lower": -1.0, "upper": 1.0}),
    "slcp":           ("box", {"lower": -3.0, "upper": 3.0}),
}
METHODS = ["PRISM", "cINN", "NPE-CNF", "VI"]
SEEDS = [0, 1, 2, 3, 4]


def _build(method, task, cfg, transform):
    if method == "PRISM":
        return make_method("PRISM", task, cfg)                 # transform is internal
    base = make_method(method, task, cfg)
    return SupportConstrained(base, transform)                 # identical transform


def run(method, task, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    kind, kw = CONSTRAINED[task]
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    transform = make_transform(kind, dim, **kw)

    est = _build(method, task, cfg, transform)
    t0 = time.time(); est.fit(ds.X_train, ds.y_train); train_s = time.time() - t0

    # posterior at held-out observations
    y0 = ds.y_test[0]
    t0 = time.time(); post = np.asarray(est.predict_posterior(y0, 1000)); lat = (time.time() - t0) / 1000 * 1e3

    # violations (should be ~0 for ALL wrapped methods now)
    viol = est.constraint_violation_rate(y0, 1000) if hasattr(est, "constraint_violation_rate") \
        else float(np.mean(~_feasible(post, kind, kw)))

    # inverse error (posterior mean vs truth)
    x_hat = post.reshape(-1, dim).mean(0)
    rmse = float(np.sqrt(np.mean((x_hat - ds.X_test[0]) ** 2)))

    # calibration vs reference where available
    metrics = {"violation_rate": viol, "inverse_rmse": rmse,
               "latency_ms": lat, "train_s": train_s}
    ref = emcee_reference(task, y0) if task in ("slcp",) else None
    if ref is not None:
        metrics["c2st"] = float(c2st(post.reshape(-1, dim), ref))
    metrics["cov90"] = float(coverage(post.reshape(-1, dim), ds.X_test[0], level=0.9))
    metrics["nll_phys"] = float(gaussian_nll(post.reshape(-1, dim), ds.X_test[0]))
    return metrics


def _feasible(x, kind, kw):
    if kind == "positive":
        return (x > 0).all(-1)
    if kind == "box":
        return ((x > kw["lower"]) & (x < kw["upper"])).all(-1)
    return np.ones(len(x), bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--manifest", default="results/manifest.json")
    args = ap.parse_args()
    cfg = load_cfg("phase2", args.mode)

    import os
    man = ResultManifest(args.manifest)
    if os.path.exists(args.manifest):
        man.load()
    for task in CONSTRAINED:
        for method in METHODS:
            man.run_seeds("phase2_constraints", task, method,
                          fn=lambda s, m=method, t=task: run(m, t, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
            v = man.agg("phase2_constraints", task, method, "violation_rate")[0]
            print(f"  [{task:16s}] {method:8s} viol={100*v:5.1f}%")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
