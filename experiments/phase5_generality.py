"""PRISM Phase 5 -- E14: cross-domain generality, raw metrics (guide item 18).

The old table used an arbitrary "inverse R2 > 0.5 and zero violations" pass/fail,
which is meaningless on unconstrained tasks and mislabels calibrated multimodal
posteriors as failures. We report RAW metrics per task under the single fixed
configuration, and select a posterior-appropriate PRIMARY metric per task
(point R2 for point tasks, a calibration metric for the disconnected multimodal
task). No binary pass/fail column.

Run in D:\\ICLR:  py experiments\\phase5_generality.py --mode smoke
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
from experiments.phase5_analysis import primary_metric
from prism.posterior.calibration import c2st

TASKS = ["burgers", "darcy", "gaussian_linear", "helmholtz",
         "lotka_volterra", "oscillator", "slcp", "two_moons"]
SEEDS = [0, 1, 2, 3, 4]


def _r2(x_hat, x):
    return float(1 - np.sum((x - x_hat) ** 2) / (np.sum((x - x.mean(0)) ** 2) + 1e-12))


def run(task, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    est = make_method("PRISM", task, cfg)
    est.fit(ds.X_train, ds.y_train)

    x_hat = np.asarray([np.asarray(est.invert(ds.y_test[i:i + 1]) if hasattr(est, "invert")
                                   else est.predict(ds.y_test[i:i + 1]))[0]
                        for i in range(len(ds.y_test))])
    fwd = np.asarray(est.predict_forward(ds.X_test) if hasattr(est, "predict_forward")
                     else est.predict(ds.X_test))
    post = np.asarray(est.predict_posterior(ds.y_test[0], 1000)).reshape(-1, dim)
    feas = est.constraint_violation_rate(ds.y_test[0], 1000) if hasattr(est, "constraint_violation_rate") else 0.0

    out = {
        "inverse_r2": _r2(x_hat, np.asarray(ds.X_test)),
        "forward_relL2": float(np.linalg.norm(fwd - ds.y_test) / (np.linalg.norm(ds.y_test) + 1e-12)),
        "violation_rate": float(feas),
        "primary_metric": primary_metric(task),
    }
    if task in ("slcp", "gaussian_linear", "two_moons"):
        ref = emcee_reference(task, ds.y_test[0]) if task != "two_moons" else None
        if ref is not None:
            out["c2st"] = float(c2st(post, ref))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--manifest", default="results/manifest.json")
    args = ap.parse_args()
    cfg = load_cfg("phase5", args.mode)

    import os
    man = ResultManifest(args.manifest)
    if os.path.exists(args.manifest):
        man.load()
    for task in TASKS:
        man.run_seeds("phase5_generality", task, "PRISM",
                      fn=lambda s, t=task: run(t, s, cfg),
                      seeds=SEEDS, config={"mode": args.mode})
        r = man.agg("phase5_generality", task, "PRISM", "inverse_r2")[0]
        print(f"  [{task:16s}] inverse R2={r:+.3f}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
