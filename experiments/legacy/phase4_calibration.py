"""Phase 4 — calibration / uncertainty quantification on SBI tasks.

Produces T4 (C2ST, coverage@90, CRPS, NLL, Wasserstein vs reference for PRISM
and posterior baselines) and the arrays behind F5 (posterior samples vs
reference), F6 (SBC rank histograms + reliability curve), F7 (coverage vs
nominal across tasks).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import numpy as np

from benchmarks.registry import get_dataset
from prism.posterior import calibration as cal
from _common import (load_cfg, make_method, fit_timed, save_results, banner)

# task -> reference kind
TASKS = [("gaussian_linear", "analytic"), ("slcp", "emcee"), ("two_moons", None)]
METHODS = ["PRISM", "NPE-CNF", "VI"]
LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95]


def _reference(ds, Y, n, kind, max_obs, seed=0):
    """Reference posterior samples (M', n, d) for the first max_obs obs, or None."""
    if kind is None:
        return None
    refs = []
    for i in range(min(max_obs, len(Y))):
        refs.append(ds.reference_posterior(Y[i], n=n, seed=seed + i))
    return np.stack(refs)


def run(mode="smoke"):
    cfg = load_cfg("phase4", mode)
    scalars = {"mode": mode, "config": cfg, "T4": {}}
    arrays = {}
    banner(f"PHASE 4 — calibration / UQ [{mode}]")
    M = cfg["n_calib_obs"]
    S = cfg["posterior_samples"]
    n_c2st = cfg["n_mcmc_obs"]                     # obs used for C2ST/W-dist

    for task, refkind in TASKS:
        ds = get_dataset(task, n=cfg["n_train"] + max(M, cfg["n_test"]), seed=0,
                         test=max(M, cfg["n_test"]) / (cfg["n_train"] + max(M, cfg["n_test"])))
        Yt, Xt = ds.Y_test[:M], ds.X_test[:M]
        ref = _reference(ds, Yt, S, refkind, n_c2st)
        scalars["T4"][task] = {}
        for name in METHODS:
            m, _ = fit_timed(make_method(name, cfg), ds.X_train, ds.Y_train)
            post = m.predict_posterior(Yt, n_samples=S)
            summ = cal.summarize(Xt, post)
            row = {"coverage90": float(cal.coverage(Xt, post)[0.9]),
                   "expected_coverage_error": summ["expected_coverage_error"],
                   "crps": summ["crps"], "nll": summ["gaussian_nll"],
                   "sbc_ks": summ["sbc_ks"]}
            if ref is not None:
                row["c2st"] = float(np.mean([cal.c2st(ref[i], post[i]) for i in range(n_c2st)]))
                row["wdist"] = float(np.mean([cal.wasserstein_to_reference(post[i], ref[i])
                                              for i in range(n_c2st)]))
            scalars["T4"][task][name] = row
            print(f"  [T4] {task:16s} {name:8s} "
                  f"cov90={row['coverage90']:.3f} ece={row['expected_coverage_error']:.3f} "
                  f"crps={row['crps']:.3f}" + (f" c2st={row.get('c2st'):.3f}" if "c2st" in row else ""))

            # arrays for figures (PRISM only, to keep the npz small)
            if name == "PRISM":
                ranks, _ = cal.sbc_ranks(Xt, post)
                arrays[f"F6_{task}_sbc_ranks"] = ranks
                cov = cal.coverage(Xt, post, levels=LEVELS)
                arrays[f"F7_{task}_levels"] = np.array(LEVELS)
                arrays[f"F7_{task}_emp"] = np.array([cov[l] for l in LEVELS])
                arrays[f"F5_{task}_prism"] = post[0]
                if ref is not None:
                    arrays[f"F5_{task}_ref"] = ref[0]
                arrays[f"F5_{task}_true"] = Xt[0]

    paths = save_results("phase4", scalars, arrays)
    print("\nsaved:", *paths)
    return scalars


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run("full" if a.full else "smoke")
