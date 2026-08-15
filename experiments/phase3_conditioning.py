"""PRISM Phase 3 -- E5: conditioning ablation (guide item 13).

The manuscript claimed dynamics-only conditioning "collapses to the prior" as if
it were a generic optimization fact. We reframe it as a controlled, measured
observation by comparing three variants under identical training:

    base      -- observation conditions the base distribution only (default PRISM)
    dynamics  -- observation conditions the velocity field only
    both      -- observation conditions base AND velocity

and reporting dependence-sensitive diagnostics: C2ST to reference, coverage,
physical NLL, the posterior spread ratio (collapse -> ~1 or larger), and the
y-dependence of the posterior mean (collapse -> ~0).

Run in D:\\ICLR:  py experiments\\phase3_conditioning.py --mode smoke
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from experiments.provenance import ResultManifest
from experiments._common import load_cfg
from experiments.api import get_dataset, emcee_reference
from experiments.phase3_variants import make_variant
from experiments.phase3_analysis import spread_ratio, y_dependence
from prism.posterior.calibration import c2st, coverage, gaussian_nll

TASKS = ["slcp", "two_moons", "gaussian_linear"]
VARIANTS = {
    "base":     dict(cond_base=True,  cond_velocity=False),
    "dynamics": dict(cond_base=False, cond_velocity=True),
    "both":     dict(cond_base=True,  cond_velocity=True),
}
SEEDS = [0, 1, 2, 3, 4]


def run(task, variant, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    prior_std = np.asarray(ds.X_train).std(0)

    est = make_variant(task, cfg, **VARIANTS[variant])
    est.fit(ds.X_train, ds.y_train)

    y0 = ds.y_test[0]
    post = np.asarray(est.predict_posterior(y0, 1000)).reshape(-1, dim)

    # posterior means across many observations -> y-dependence (collapse if ~0)
    n_obs = min(32, len(ds.y_test))
    means = np.stack([
        np.asarray(est.predict_posterior(ds.y_test[i], 200)).reshape(-1, dim).mean(0)
        for i in range(n_obs)
    ], 0)

    out = {
        "cov90": float(coverage(post, ds.X_test[0], level=0.9)),
        "nll_phys": float(gaussian_nll(post, ds.X_test[0])),
        "spread_ratio": spread_ratio(post, prior_std),
        "y_dependence": y_dependence(means, prior_std),
    }
    ref = emcee_reference(task, y0) if task in ("slcp", "gaussian_linear") else None
    if ref is not None:
        out["c2st"] = float(c2st(post, ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--manifest", default="results/manifest.json")
    args = ap.parse_args()
    cfg = load_cfg("phase3", args.mode)

    import os
    man = ResultManifest(args.manifest)
    if os.path.exists(args.manifest):
        man.load()
    for task in TASKS:
        for variant in VARIANTS:
            man.run_seeds("phase3_conditioning", task, variant,
                          fn=lambda s, v=variant, t=task: run(t, v, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        for v in VARIANTS:
            yd = man.agg("phase3_conditioning", task, v, "y_dependence")[0]
            sr = man.agg("phase3_conditioning", task, v, "spread_ratio")[0]
            print(f"  [{task:16s}] {v:9s} y-dep={yd:.3f} spread={sr:.2f}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
