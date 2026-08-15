"""Phase 3 — operator scaling / zero-shot super-resolution.

Trains forward operators at a base field resolution and evaluates their error
when queried at higher resolutions (2x, 4x) without retraining. Produces T5
(error at train-res / 2x / 4x) and F8 (error-vs-resolution curve). FNO is
resolution-agnostic by construction; the MLP-based operators are included for
contrast where applicable.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import numpy as np

from benchmarks.registry import get_dataset
from baselines._common import fields_from_dataset
from baselines import FNO
from _common import load_cfg, relative_l2, save_results, banner


def _resample_field(field, factor):
    """Up-sample a 1D/2D field by an integer factor via linear interpolation."""
    field = np.asarray(field, float)
    if field.ndim == 1:
        n = field.shape[0]
        xs = np.linspace(0, 1, n)
        xt = np.linspace(0, 1, n * factor)
        return np.interp(xt, xs, field)
    else:
        from scipy.ndimage import zoom
        return zoom(field, factor, order=1)


def _fields_at(ds, X, factor):
    base = fields_from_dataset(ds, X)
    if factor == 1:
        return base
    return np.stack([_resample_field(f, factor) for f in base])


def run(mode="smoke"):
    cfg = load_cfg("phase3", mode)
    scalars = {"mode": mode, "config": cfg, "T5": {}}
    arrays = {}
    banner(f"PHASE 3 — operator scaling [{mode}]")
    factors = [1, 2, 4]

    for task in ["burgers", "helmholtz", "darcy"]:
        ds = get_dataset(task, n=cfg["n_train"] + cfg["n_test"], seed=0,
                         test=cfg["n_test"] / (cfg["n_train"] + cfg["n_test"]))
        Ftr = fields_from_dataset(ds, ds.X_train)
        fno = FNO(epochs=cfg["epochs"], seed=0).fit(Ftr, ds.Y_train)
        errs = {}
        for fac in factors:
            Fte = _fields_at(ds, ds.X_test, fac)
            pred = fno.predict(Fte)
            errs[f"{fac}x"] = relative_l2(ds.Y_test, pred)
        scalars["T5"][task] = errs
        arrays[f"F8_{task}_factors"] = np.array(factors)
        arrays[f"F8_{task}_relL2"] = np.array([errs[f"{f}x"] for f in factors])
        print(f"  [T5] {task:10s} " + "  ".join(f"{k}={v:.4f}" for k, v in errs.items()))

    paths = save_results("phase3", scalars, arrays)
    print("\nsaved:", *paths)
    return scalars


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run("full" if a.full else "smoke")
