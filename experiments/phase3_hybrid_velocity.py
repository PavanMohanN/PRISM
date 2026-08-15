"""PRISM Phase 3 -- E6: hybrid conditional velocity (guide item 14).

In default PRISM the observation y modifies only the base; the transport F is
shared across all y. This may limit how much posterior SHAPE can vary with y. We
test a hybrid that also conditions the velocity field, on one simple unimodal
task and one shape-changing / harder task, with matched parameter budgets. If the
hybrid brings no benefit we report that as evidence the simpler shared transport
suffices -- an honest negative result rather than a hidden design choice.

Run in D:\\ICLR:  py experiments\\phase3_hybrid_velocity.py --mode smoke
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
from experiments.phase3_variants import make_variant, count_params
from prism.posterior.calibration import c2st, coverage, gaussian_nll

TASKS = ["gaussian_linear", "slcp"]     # one unimodal, one shape-changing
VARIANTS = {
    "shared": dict(cond_base=True, cond_velocity=False),
    "hybrid": dict(cond_base=True, cond_velocity=True),
}
SEEDS = [0, 1, 2, 3, 4]


def run(task, variant, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    est = make_variant(task, cfg, **VARIANTS[variant])
    est.fit(ds.X_train, ds.y_train)

    y0 = ds.y_test[0]
    post = np.asarray(est.predict_posterior(y0, 1000)).reshape(-1, dim)
    out = {
        "cov90": float(coverage(post, ds.X_test[0], level=0.9)),
        "nll_phys": float(gaussian_nll(post, ds.X_test[0])),
        "n_params": count_params(est),
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
            man.run_seeds("phase3_hybrid", task, variant,
                          fn=lambda s, v=variant, t=task: run(t, v, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        cs_share = man.agg("phase3_hybrid", task, "shared", "c2st") if task != "two_moons" else (float("nan"),)
        print(f"  [{task}] shared vs hybrid recorded")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
