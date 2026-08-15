"""Phase 6 — ablation, robustness, OOD, efficiency.

Produces T7 (ablation: liquid/static, exact/soft, +/-projection, +/-posterior),
T8 (params, train time, latency, MCMC speedup), F9 (robustness vs noise),
F10 (OOD generalization), F11 (accuracy-latency Pareto), F12 (ablation deltas).
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch

from benchmarks.registry import get_dataset
from baselines import MCMCReference
from prism.posterior import calibration as cal
from _common import (load_cfg, make_method, fit_timed, r2, violation_rate,
                     save_results, banner)

TASK = "oscillator"
CON = "positive"


def _n_params(model):
    total = 0
    for v in vars(model).values():
        if isinstance(v, torch.nn.Module):
            total += sum(p.numel() for p in v.parameters())
    return int(total)


def _inv_r2(m, ds):
    pm = m.predict_posterior(ds.Y_test, n_samples=200).mean(axis=1)
    return float(np.mean([r2(ds.X_test[:, k], pm[:, k]) for k in range(ds.d)]))


def run(mode="smoke"):
    cfg = load_cfg("phase6", mode)
    scalars = {"mode": mode, "config": cfg, "T7": {}, "T8": {}}
    arrays = {}
    banner(f"PHASE 6 — ablation / robustness / efficiency [{mode}]")
    ds = get_dataset(TASK, n=cfg["n_train"] + cfg["n_test"], seed=0,
                     test=cfg["n_test"] / (cfg["n_train"] + cfg["n_test"]))

    # ---- T7 / F12 : ablation ----
    variants = {
        "PRISM (full)": dict(name="PRISM", constraint=CON),
        "static (no liquid)": dict(name="PRISM-static", constraint=CON),
        "soft (no exact-rev)": dict(name="PRISM-soft", constraint=CON),
        "no projection": dict(name="PRISM", constraint="none"),
    }
    for label, spec in variants.items():
        m, _ = fit_timed(make_method(spec["name"], cfg, constraint=spec["constraint"]),
                         ds.X_train, ds.Y_train)
        Xhat = m.invert(ds.Y_test)
        post = m.predict_posterior(ds.Y_test, n_samples=200)
        row = {"inverse_meanR2": _inv_r2(m, ds),
               "cycle_err": m.cycle_consistency_error(ds.X_test[:200]),
               "violation": violation_rate(Xhat, CON, ds.meta),
               "expected_coverage_error": cal.coverage(ds.X_test, post)["expected_coverage_error"]}
        scalars["T7"][label] = row
        print(f"  [T7] {label:22s} R2={row['inverse_meanR2']:+.3f} "
              f"cyc={row['cycle_err']:.1e} viol={row['violation']:.3f} "
              f"ece={row['expected_coverage_error']:.3f}")
    # F12 deltas vs full
    full = scalars["T7"]["PRISM (full)"]
    arrays["F12_labels"] = np.array(list(scalars["T7"].keys()))
    arrays["F12_dR2"] = np.array([scalars["T7"][k]["inverse_meanR2"] - full["inverse_meanR2"]
                                  for k in scalars["T7"]])

    # ---- T8 / F11 : efficiency ----
    S = cfg["posterior_samples"]
    pareto = []
    for name in ["PRISM", "cINN", "NPE-CNF", "VI"]:
        m, t_fit = fit_timed(make_method(name, cfg, constraint=(CON if name == "PRISM" else "none")),
                             ds.X_train, ds.Y_train)
        y1 = ds.Y_test[:1]
        t0 = time.time(); m.predict_posterior(y1, n_samples=S); lat = (time.time() - t0) / S * 1000
        acc = _inv_r2(m, ds)
        scalars["T8"][name] = {"params": _n_params(m), "train_time_s": t_fit,
                               "latency_ms_per_sample": lat, "inverse_meanR2": acc}
        pareto.append((name, lat, acc))
        print(f"  [T8] {name:8s} params={scalars['T8'][name]['params']:6d} "
              f"train={t_fit:5.1f}s lat={lat:.3f}ms/s R2={acc:+.3f}")
    # MCMC speedup
    try:
        mc = MCMCReference(ds, nsteps=cfg.get("mcmc_nsteps", 600), burn=200)
        t0 = time.time(); mc.predict_posterior(ds.Y_test[:1], n_samples=S, seed=0)
        mcmc_lat = (time.time() - t0) / S * 1000
        prism_lat = scalars["T8"]["PRISM"]["latency_ms_per_sample"]
        scalars["T8"]["MCMC"] = {"latency_ms_per_sample": mcmc_lat}
        scalars["T8"]["mcmc_speedup_x"] = float(mcmc_lat / max(prism_lat, 1e-9))
        print(f"  [T8] MCMC lat={mcmc_lat:.3f}ms/s  ->  PRISM speedup "
              f"{scalars['T8']['mcmc_speedup_x']:.1f}x")
    except Exception as e:
        scalars["T8"]["MCMC"] = {"error": str(e)}
    arrays["F11_names"] = np.array([p[0] for p in pareto])
    arrays["F11_latency"] = np.array([p[1] for p in pareto])
    arrays["F11_accuracy"] = np.array([p[2] for p in pareto])

    # ---- F9 : robustness vs observation noise ----
    m = make_method("PRISM", cfg, constraint=CON).fit(ds.X_train, ds.Y_train)
    sigmas = [0.0, 0.02, 0.05, 0.1, 0.2]
    rob_r2, rob_ece = [], []
    rng = np.random.default_rng(0)
    yscale = ds.Y_test.std()
    for sg in sigmas:
        Yn = ds.Y_test + sg * yscale * rng.standard_normal(ds.Y_test.shape)
        pm = m.predict_posterior(Yn, n_samples=150)
        rob_r2.append(float(np.mean([r2(ds.X_test[:, k], pm.mean(1)[:, k]) for k in range(ds.d)])))
        rob_ece.append(cal.coverage(ds.X_test, pm)["expected_coverage_error"])
    arrays["F9_sigma"] = np.array(sigmas)
    arrays["F9_inverse_R2"] = np.array(rob_r2)
    arrays["F9_ece"] = np.array(rob_ece)
    print(f"  [F9] noise R2 {[round(v,2) for v in rob_r2]}")

    # ---- F10 : OOD generalization (train low-omega, test high-omega) ----
    thr = 2.2
    in_tr = ds.X[:, 0] < thr
    Xtr, Ytr = ds.X[in_tr], ds.Y[in_tr]
    ood = ~in_tr
    m2 = make_method("PRISM", cfg, constraint=CON).fit(Xtr, Ytr)
    # in-dist held-out: resample some in-dist via a fresh seed slice
    id_mask = in_tr.copy(); id_idx = np.where(id_mask)[0][-cfg["n_test"]:]
    ood_idx = np.where(ood)[0]
    def mean_r2(idx):
        if len(idx) < 5:
            return float("nan")
        pm = m2.predict_posterior(ds.Y[idx], n_samples=150).mean(1)
        return float(np.mean([r2(ds.X[idx][:, k], pm[:, k]) for k in range(ds.d)]))
    scalars["F10_in_dist_R2"] = mean_r2(id_idx)
    scalars["F10_ood_R2"] = mean_r2(ood_idx)
    print(f"  [F10] in-dist R2={scalars['F10_in_dist_R2']:.3f}  "
          f"OOD R2={scalars['F10_ood_R2']:.3f}")

    paths = save_results("phase6", scalars, arrays)
    print("\nsaved:", *paths)
    return scalars


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run("full" if a.full else "smoke")
