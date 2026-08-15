"""PRISM Phase 2 -- E1: reversibility sweep (guide items 1, 4, 5, 21).

Fixes the "flat invertibility curve" objection by:
  * sweeping a BROAD step range {2,4,8,16,32,64,128,256},
  * integrating in float64 so the O(h^p) regime is visible before the floor,
  * measuring the LATENT cycle (guaranteed) and the PHYSICAL cycle (empirical)
    SEPARATELY, so we never conflate the two again.

Writes one record per (task, seed) to the shared manifest; the reversibility
curve is stored as arrays for the figure, and the fitted order/floor as scalars
for the table.

Run in D:\\ICLR:  py experiments\\phase2_reversibility.py --mode smoke
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from experiments.provenance import ResultManifest
from experiments._common import load_cfg
from experiments.api import make_method          # existing repo helpers
from experiments.api import get_dataset
from prism.utils.metrics_cycle import latent_cycle_error, physical_cycle_error
from experiments.phase2_analysis import loglog_order_and_floor

STEPS = [2, 4, 8, 16, 32, 64, 128, 256]
TASKS = ["oscillator", "darcy", "burgers"]      # 1 dynamical + 2 PDE
SEEDS = [0, 1, 2, 3, 4]


# ----- repo hooks (adjust attribute names to your codebase if needed) -----
def _set_solver_steps(prism, k: int):
    """Set the fixed-step count of the liquid-ODE flow."""
    flow = getattr(prism, "flow_", getattr(prism, "flow", None))
    for attr in ("n_steps", "num_steps", "steps"):
        if hasattr(flow, attr):
            setattr(flow, attr, k); return
    raise AttributeError("could not find the flow's step-count attribute; set it in _set_solver_steps")


def _flow(prism):
    return getattr(prism, "flow_", getattr(prism, "flow", None))


def _forward_fn(prism):
    # auxiliary physical forward surrogate g_psi:  x -> y
    fn = getattr(prism, "predict_forward", None) or getattr(prism, "forward_predict", None)
    if fn is None:
        raise AttributeError("expose the forward surrogate as prism.predict_forward(x)")
    return lambda X: torch.as_tensor(fn(X), dtype=torch.float64)


def _inverse_fn(prism):
    # amortized inverse path h:  y -> x  (point estimate, incl. projection)
    fn = getattr(prism, "invert", None) or getattr(prism, "predict", None)
    return lambda Y: torch.as_tensor(fn(Y), dtype=torch.float64)


def run_task(task: str, seed: int, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_default_dtype(torch.float64)              # expose convergence regime

    ds = get_dataset(task, seed=seed)
    prism = make_method("PRISM", task, cfg)
    prism.fit(ds.X_train, ds.y_train)

    X = torch.as_tensor(ds.X_test, dtype=torch.float64)
    Xt = prism.transform_.inverse(X) if hasattr(prism, "transform_") else X

    lat_curve, phys_curve = [], []
    for k in STEPS:
        _set_solver_steps(prism, k)
        lat = latent_cycle_error(_flow(prism), Xt)["latent_cycle_mean"]
        phys = physical_cycle_error(_forward_fn(prism), _inverse_fn(prism), X)["physical_cycle_mean"]
        lat_curve.append(lat); phys_curve.append(phys)

    lat_fit = loglog_order_and_floor(STEPS, lat_curve)
    return {
        "steps": list(STEPS),
        "latent_cycle_curve": lat_curve,
        "physical_cycle_curve": phys_curve,
        "latent_order": lat_fit["order"],
        "latent_floor": lat_fit["floor"],
        "physical_cycle_mean": float(np.mean(phys_curve)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--manifest", default="results/manifest.json")
    args = ap.parse_args()
    cfg = load_cfg("phase2", args.mode)

    man = ResultManifest(args.manifest)
    import os
    if os.path.exists(args.manifest):
        man.load()
    for task in TASKS:
        man.run_seeds("phase2_reversibility", task, "PRISM",
                      fn=lambda s, t=task: run_task(t, s, cfg),
                      seeds=SEEDS, config={"mode": args.mode, "steps": STEPS})
        m, s, n = man.agg("phase2_reversibility", task, "PRISM", "latent_order")
        print(f"  [{task}] latent order = {m:.2f} +/- {s:.2f}  (n={n})")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
