"""PRISM Phase 3 -- E4: liquid vs static across tasks (guide items 15, 30).

One oscillator result cannot justify the "liquid" novelty. We run liquid vs
static on FIVE representative tasks with MATCHED parameter counts and 5 seeds,
recording inverse accuracy, calibration, and the parameter count itself so the
comparison is auditable. The table verdict (helps/neutral/hurts) is a paired,
sign-consistent, effect-size test across seeds -- not a single-run gap.

Run in D:\\ICLR:  py experiments\\phase3_liquid_vs_static.py --mode smoke
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

TASKS = ["darcy", "slcp", "two_moons", "oscillator", "lotka_volterra"]
VARIANTS = {"liquid": dict(liquid=True), "static": dict(liquid=False)}
SEEDS = [0, 1, 2, 3, 4]


def _r2(x_hat, x):
    ss_res = np.sum((x - x_hat) ** 2)
    ss_tot = np.sum((x - x.mean(0)) ** 2) + 1e-12
    return float(1 - ss_res / ss_tot)


def run(task, variant, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    est = make_variant(task, cfg, **VARIANTS[variant])
    est.fit(ds.X_train, ds.y_train)

    y0 = ds.y_test[0]
    post = np.asarray(est.predict_posterior(y0, 1000)).reshape(-1, dim)
    x_hat = post.mean(0)

    out = {
        "inverse_r2": _r2(np.asarray([est.invert(torch.as_tensor(y[None]))[0].numpy()
                                      if hasattr(est, "invert") else est.predict(y[None])[0]
                                      for y in ds.y_test]),
                          np.asarray(ds.X_test)),
        "cov90": float(coverage(post, ds.X_test[0], level=0.9)),
        "nll_phys": float(gaussian_nll(post, ds.X_test[0])),
        "n_params": count_params(est),
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
    for task in TASKS:
        for variant in VARIANTS:
            man.run_seeds("phase3_liquid", task, variant,
                          fn=lambda s, v=variant, t=task: run(t, v, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        rl = man.agg("phase3_liquid", task, "liquid", "inverse_r2")[0]
        rs = man.agg("phase3_liquid", task, "static", "inverse_r2")[0]
        print(f"  [{task:16s}] liquid R2={rl:.3f}  static R2={rs:.3f}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
