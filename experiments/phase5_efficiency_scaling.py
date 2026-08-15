"""PRISM Phase 5 -- E12: efficiency + amortization scaling (guide items 20, 26).

Two parts:
  (a) Efficiency: parameters, training time, and per-sample latency for each
      method on a base task. The accuracy--latency figure is drawn from THESE
      SAME records, so the figure caption can never disagree with the table
      (item 26).
  (b) Amortization scaling: for a fixed effective-sample target, we measure the
      amortized total cost (train once + cheap sampling) against a per-observation
      reference (MCMC) as the parameter dimension grows, and report the number of
      test observations M* beyond which amortization wins in total wall-clock. This
      replaces the unsupported "0.3x speedup" with a demonstrated crossover
      (item 20).

Run in D:\\ICLR:  py experiments\\phase5_efficiency_scaling.py --mode smoke
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
from experiments.api import get_dataset, mcmc_reference
from experiments.phase5_analysis import crossover_observations

BASE_TASK = "oscillator"
METHODS = ["PRISM", "cINN", "NPE-CNF", "VI"]
DIMS = [2, 8, 32]              # scaling dimension for the crossover study
N_SAMPLES = 1000               # effective-sample target per observation
SEEDS = [0, 1, 2, 3, 4]


def run_efficiency(method, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(BASE_TASK, seed=seed)
    dim = ds.X_train.shape[-1]
    est = make_method(method, BASE_TASK, cfg)
    t0 = time.time(); est.fit(ds.X_train, ds.y_train); train_s = time.time() - t0
    n_params = sum(p.numel() for m in vars(est).values()
                   if isinstance(m, torch.nn.Module) for p in m.parameters())
    y0 = ds.y_test[0]
    t0 = time.time(); post = np.asarray(est.predict_posterior(y0, N_SAMPLES)); lat = (time.time() - t0) / N_SAMPLES * 1e3
    x_hat = post.reshape(-1, dim).mean(0)
    ss = 1 - np.sum((ds.X_test[0] - x_hat) ** 2) / (np.sum((ds.X_test[0] - ds.X_test[0].mean()) ** 2) + 1e-12)
    return {"n_params": int(n_params), "train_s": float(train_s),
            "latency_ms": float(lat), "inverse_r2": float(ss)}


def run_scaling(seed, cfg) -> dict:
    """Measure amortized vs MCMC per-target cost as dimension grows -> crossover."""
    torch.manual_seed(seed); np.random.seed(seed)
    dims, m_star = [], []
    amort_per_sample, ref_per_target, train_costs = [], [], []
    for d in DIMS:
        ds = get_dataset("gaussian_linear", seed=seed, dim=d)
        est = make_method("PRISM", "gaussian_linear", cfg)
        t0 = time.time(); est.fit(ds.X_train, ds.y_train); train = time.time() - t0
        y0 = ds.y_test[0]
        t0 = time.time(); est.predict_posterior(y0, N_SAMPLES); ps = (time.time() - t0) / N_SAMPLES
        t0 = time.time(); mcmc_reference("gaussian_linear", y0, n_samples=N_SAMPLES); ref = time.time() - t0
        ms = crossover_observations(train, ps, N_SAMPLES, ref)
        dims.append(d); m_star.append(ms)
        amort_per_sample.append(ps); ref_per_target.append(ref); train_costs.append(train)
    return {"dims": dims, "crossover_M": m_star,
            "amort_per_sample": amort_per_sample, "ref_per_target": ref_per_target,
            "train_costs": train_costs}


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
    for m in METHODS:
        man.run_seeds("phase5_efficiency", BASE_TASK, m,
                      fn=lambda s, mm=m: run_efficiency(mm, s, cfg),
                      seeds=SEEDS, config={"mode": args.mode})
    man.run_seeds("phase5_scaling", "gaussian_linear", "PRISM",
                  fn=lambda s: run_scaling(s, cfg), seeds=SEEDS,
                  config={"mode": args.mode, "dims": DIMS})
    rec = [r for k, r in man.records.items() if k[:3] == ("phase5_scaling", "gaussian_linear", "PRISM")][0]
    print(f"  crossover M* by dim {DIMS}: {[round(x,1) for x in rec.metrics['crossover_M']]}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
