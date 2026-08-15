"""PRISM Phase 5 -- E11: main PDE accuracy (guide items 16, 29).

Forward (relative L2) and inverse (RMSE) on the three PDE tasks, 5 seeds, so the
table carries mean +/- std. The forward block honestly includes the dedicated
operators that beat PRISM; the text uses the numeric gap (relative_gap) rather
than vague "competitive" wording. One field example per task is saved for the
recovery figure so it is provably PRISM's output.

Run in D:\\ICLR:  py experiments\\phase5_main_pde.py --mode smoke
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
from experiments.api import get_dataset

TASKS = ["darcy", "burgers", "helmholtz"]
FWD = ["PRISM", "FNO", "DeepONet", "PINN"]
INV = ["PRISM", "cINN", "NPE-CNF", "VI"]
SEEDS = [0, 1, 2, 3, 4]


def _relL2(pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def _rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def run_forward(task, method, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    est = make_method(method, task, cfg)
    est.fit(ds.X_train, ds.y_train)
    pred = np.asarray(est.predict_forward(ds.X_test) if hasattr(est, "predict_forward")
                      else est.predict(ds.X_test))
    out = {"forward_relL2": _relL2(pred, ds.y_test)}
    if method == "PRISM" and task == "darcy":     # save one field example for the figure
        out["field_true"] = np.asarray(ds.X_test[0]).tolist()
        inv = est.invert(ds.y_test[:1]) if hasattr(est, "invert") else est.predict(ds.y_test[:1])
        out["field_pred"] = np.asarray(inv[0]).tolist()
    return out


def run_inverse(task, method, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    est = make_method(method, task, cfg)
    est.fit(ds.X_train, ds.y_train)
    x_hat = np.asarray([np.asarray(est.invert(ds.y_test[i:i + 1]) if hasattr(est, "invert")
                                   else est.predict(ds.y_test[i:i + 1]))[0]
                        for i in range(len(ds.y_test))])
    return {"inverse_rmse": _rmse(x_hat, ds.X_test)}


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
        for m in FWD:
            man.run_seeds("phase5_pde_fwd", task, m,
                          fn=lambda s, mm=m, t=task: run_forward(t, mm, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        for m in INV:
            man.run_seeds("phase5_pde_inv", task, m,
                          fn=lambda s, mm=m, t=task: run_inverse(t, mm, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        pr = man.agg("phase5_pde_inv", task, "PRISM", "inverse_rmse")[0]
        print(f"  [{task:10s}] PRISM inverse RMSE={pr:.3f}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
