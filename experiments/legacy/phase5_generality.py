"""Phase 5 — cross-domain generality.

Runs a *single, fixed* PRISM configuration across every benchmark family (PDE,
SBI, dynamics) and records a key metric plus a pass/fail for each task. This
backs T6: one model, no per-task tuning, competent everywhere.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import numpy as np

from benchmarks.registry import get_dataset, list_datasets
from _common import (load_cfg, make_method, fit_timed, rmse, r2, relative_l2,
                     violation_rate, save_results, banner)

# fixed constraint per task (physics), but identical PRISM hyper-params throughout
CONSTRAINT = {"oscillator": "positive", "lotka_volterra": "positive",
              "two_moons": "box", "slcp": "box"}
PASS_R2 = 0.5                          # inverse posterior-mean R^2 threshold


def run(mode="smoke"):
    cfg = load_cfg("phase5", mode)
    scalars = {"mode": mode, "config": cfg, "T6": {}}
    banner(f"PHASE 5 — cross-domain generality [{mode}]")

    n_pass = 0
    for task in list_datasets():
        con = CONSTRAINT.get(task, "none")
        ds = get_dataset(task, n=cfg["n_train"] + cfg["n_test"], seed=0,
                         test=cfg["n_test"] / (cfg["n_train"] + cfg["n_test"]))
        # identical config everywhere (only the physics constraint varies)
        m, t_fit = fit_timed(make_method("PRISM", cfg, constraint=con),
                             ds.X_train, ds.Y_train)
        Xhat = m.invert(ds.Y_test)
        post = m.predict_posterior(ds.Y_test, n_samples=max(100, cfg["posterior_samples"] // 2))
        pm = post.mean(axis=1)
        inv_r2 = float(np.mean([r2(ds.X_test[:, k], pm[:, k]) for k in range(ds.d)]))
        fwd = relative_l2(ds.Y_test, m.predict(ds.X_test))
        viol = violation_rate(Xhat, con, ds.meta)
        passed = bool(inv_r2 > PASS_R2 and viol < 1e-6)
        n_pass += passed
        scalars["T6"][task] = {
            "family": ("pde" if task in ("darcy", "burgers", "helmholtz")
                       else "sbi" if task in ("gaussian_linear", "two_moons", "slcp")
                       else "dynamics"),
            "constraint": con, "inverse_meanR2": inv_r2,
            "forward_relL2": fwd, "violation": viol,
            "cycle_err": m.cycle_consistency_error(ds.X_test[:200]),
            "fit_time_s": t_fit, "pass": passed,
        }
        print(f"  [T6] {task:16s} invR2={inv_r2:+.3f} fwdL2={fwd:.3f} "
              f"viol={viol:.3f} pass={passed}")

    scalars["T6_summary"] = {"passed": n_pass, "total": len(list_datasets())}
    print(f"\n  generality: {n_pass}/{len(list_datasets())} tasks pass with one config")
    paths = save_results("phase5", scalars)
    print("saved:", *paths)
    return scalars


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run("full" if a.full else "smoke")
