"""Phase 2 — PDE core: forward/inverse accuracy, constraints, reversibility.

Produces results for T1 (forward rel-L2 + inverse RMSE on Darcy/Burgers/
Helmholtz), T2 (constraint-violation on the constrained tasks), T3 (cycle
consistency), F2 (recovered-field examples) and F13 (invertibility-vs-solver-tol
and empirical stability).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import numpy as np

from benchmarks.registry import get_dataset
from baselines._common import fields_from_dataset
from prism.theory import stability as stab
from _common import (load_cfg, make_method, fit_timed, relative_l2, rmse, r2,
                     violation_rate, save_results, banner)

PDE_TASKS = ["darcy", "burgers", "helmholtz"]
CONSTRAINED = [("oscillator", "positive"), ("lotka_volterra", "positive"),
               ("two_moons", "box"), ("slcp", "box")]
FORWARD_METHODS = ["PRISM", "FNO", "DeepONet", "PINN"]
INVERSE_METHODS = ["PRISM", "cINN", "NPE-CNF", "VI"]


def _forward_eval(name, cfg, ds, fields_tr, fields_te):
    """Return forward rel-L2 on the test split for a forward method."""
    if name == "PRISM":
        m, _ = fit_timed(make_method("PRISM", cfg), ds.X_train, ds.Y_train)
        pred = m.predict(ds.X_test)
    else:
        m, _ = fit_timed(make_method(name, cfg), fields_tr, ds.Y_train)
        pred = m.predict(fields_te)
    return relative_l2(ds.Y_test, pred)


def _inverse_eval(name, cfg, ds, constraint="none"):
    m, _ = fit_timed(make_method(name, cfg, constraint=constraint),
                     ds.X_train, ds.Y_train)
    Xhat = m.invert(ds.Y_test)
    return m, rmse(ds.X_test, Xhat), Xhat


def run(mode="smoke"):
    cfg = load_cfg("phase2", mode)
    scalars = {"mode": mode, "config": cfg, "T1": {}, "T2": {}, "T3": {}}
    arrays = {}
    banner(f"PHASE 2 — PDE core [{mode}]")

    # ---- T1 + F2 on PDE tasks ----
    for task in PDE_TASKS:
        ds = get_dataset(task, n=cfg["n_train"] + cfg["n_test"], seed=0,
                         test=cfg["n_test"] / (cfg["n_train"] + cfg["n_test"]))
        Ftr = fields_from_dataset(ds, ds.X_train)
        Fte = fields_from_dataset(ds, ds.X_test)
        scalars["T1"][task] = {"forward_relL2": {}, "inverse_rmse": {}}
        for fm in FORWARD_METHODS:
            scalars["T1"][task]["forward_relL2"][fm] = _forward_eval(fm, cfg, ds, Ftr, Fte)
            print(f"  [T1] {task:10s} forward {fm:9s} relL2={scalars['T1'][task]['forward_relL2'][fm]:.4f}")
        prism_inv = None
        for im in INVERSE_METHODS:
            m, err, Xhat = _inverse_eval(im, cfg, ds)
            scalars["T1"][task]["inverse_rmse"][im] = err
            if im == "PRISM":
                prism_inv = m
            print(f"  [T1] {task:10s} inverse {im:9s} rmse ={err:.4f}")
        # F2: one recovered-field example
        Xhat0 = prism_inv.invert(ds.Y_test[:1])[0]
        arrays[f"F2_{task}_true"] = np.asarray(ds.decode_field(ds.X_test[0]))
        arrays[f"F2_{task}_pred"] = np.asarray(ds.decode_field(Xhat0))

    # ---- T2: constraint satisfaction on constrained tasks (posterior samples) ----
    for task, con in CONSTRAINED:
        ds = get_dataset(task, n=cfg["n_train"] + cfg["n_test"], seed=0,
                         test=cfg["n_test"] / (cfg["n_train"] + cfg["n_test"]))
        scalars["T2"][task] = {"constraint": con}
        n_obs = min(60, len(ds.Y_test))
        ns = 100
        # PRISM with the hard constraint -> samples always feasible
        mp, _, _ = _inverse_eval("PRISM", cfg, ds, constraint=con)
        sp = mp.predict_posterior(ds.Y_test[:n_obs], n_samples=ns).reshape(-1, ds.d)
        scalars["T2"][task]["PRISM"] = violation_rate(sp, con, ds.meta)
        for im in ["cINN", "NPE-CNF", "VI"]:
            m, _ = fit_timed(make_method(im, cfg), ds.X_train, ds.Y_train)
            s = m.predict_posterior(ds.Y_test[:n_obs], n_samples=ns).reshape(-1, ds.d)
            scalars["T2"][task][im] = violation_rate(s, con, ds.meta)
        print(f"  [T2] {task:14s} viol%% (posterior) PRISM={scalars['T2'][task]['PRISM']:.3f} "
              f"cINN={scalars['T2'][task]['cINN']:.3f} NPE={scalars['T2'][task]['NPE-CNF']:.3f} "
              f"VI={scalars['T2'][task]['VI']:.3f}")

    # ---- T3: cycle consistency (exact vs soft vs coupling) ----
    for task in ["oscillator", "darcy"]:
        ds = get_dataset(task, n=cfg["n_train"], seed=0)
        row = {}
        for im in ["PRISM", "PRISM-soft", "cINN"]:
            m, _ = fit_timed(make_method(im, cfg), ds.X_train, ds.Y_train)
            row[im] = m.cycle_consistency_error(ds.X_train[:200])
        scalars["T3"][task] = row
        print(f"  [T3] {task:10s} cycle PRISM={row['PRISM']:.2e} "
              f"soft={row['PRISM-soft']:.2e} cINN={row['cINN']:.2e}")

    # ---- F13: invertibility vs solver tolerance + stability ----
    ds = get_dataset("oscillator", n=cfg["n_train"], seed=0)
    base = make_method("PRISM", cfg, constraint="positive")
    base.fit(ds.X_train, ds.Y_train)
    steps_grid = [2, 4, 8, 16, 32]
    cyc = []
    for ns in steps_grid:
        base.n_steps = ns                       # reuse trained flow, vary solver
        cyc.append(base.cycle_consistency_error(ds.X_train[:200]))
    arrays["F13_solver_steps"] = np.array(steps_grid)
    arrays["F13_cycle_err"] = np.array(cyc)
    scalars["F13_stability"] = stab.stability_report(base, ds.Y_test[:100])
    print(f"  [F13] cycle@steps {dict(zip(steps_grid, [f'{c:.1e}' for c in cyc])) }")
    print(f"  [F13] stability bounded={scalars['F13_stability']['bounded']}")

    paths = save_results("phase2", scalars, arrays)
    print("\nsaved:", *paths)
    return scalars


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run("full" if a.full else "smoke")
