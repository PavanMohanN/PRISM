"""PRISM Phase 3 -- E7: projection ablation on a box task (guide item 11).

The oscillator no-projection ablation was uninformative because zero violations
occurred with or without the projection. We repeat Full-PRISM vs No-Projection on
the BOX-constrained tasks (SLCP, Two Moons), where unconstrained tail samples are
expected to leave the feasible set. This isolates what the projection buys:
support satisfaction, while checking whether it changes posterior quality.

Run in D:\\ICLR:  py experiments\\phase3_projection.py --mode smoke
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
from prism.posterior.calibration import c2st, coverage, gaussian_nll

BOX = {"two_moons": (-1.0, 1.0), "slcp": (-3.0, 3.0)}
VARIANTS = {"proj": dict(use_projection=True), "noproj": dict(use_projection=False)}
SEEDS = [0, 1, 2, 3, 4]


def run(task, variant, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    lo, hi = BOX[task]
    est = make_variant(task, cfg, **VARIANTS[variant])
    est.fit(ds.X_train, ds.y_train)

    y0 = ds.y_test[0]
    post = np.asarray(est.predict_posterior(y0, 1000)).reshape(-1, dim)
    feasible = ((post > lo) & (post < hi)).all(-1)
    out = {
        "violation_rate": float(np.mean(~feasible)),
        "inverse_rmse": float(np.sqrt(np.mean((post.mean(0) - ds.X_test[0]) ** 2))),
        "cov90": float(coverage(post, ds.X_test[0], level=0.9)),
        "nll_phys": float(gaussian_nll(post, ds.X_test[0])),
    }
    ref = emcee_reference(task, y0) if task == "slcp" else None
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
    for task in BOX:
        for variant in VARIANTS:
            man.run_seeds("phase3_projection", task, variant,
                          fn=lambda s, v=variant, t=task: run(t, v, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        vp = man.agg("phase3_projection", task, "proj", "violation_rate")[0]
        vn = man.agg("phase3_projection", task, "noproj", "violation_rate")[0]
        print(f"  [{task:16s}] viol proj={100*vp:.1f}%  noproj={100*vn:.1f}%")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
