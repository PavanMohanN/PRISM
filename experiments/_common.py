"""Shared harness for the experiment phases.

Centralises: config loading (smoke vs full), construction of PRISM and all
baselines from a config, the metric helpers used across phases, and result
caching to ``results/<phase>.json`` (scalars) and ``results/<phase>.npz``
(arrays for figures). Phases stay short and declarative.
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
CONFIG_DIR = os.path.join(ROOT, "configs")
os.makedirs(RESULTS_DIR, exist_ok=True)
# make `benchmarks`, `baselines` (repo root) and sibling phase modules importable
for _p in (ROOT, os.path.dirname(os.path.abspath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------- config
# Code-level defaults; configs/<phase>.yaml may override (smoke/full sections).
_DEFAULTS = {
    "smoke": {"n_train": 600, "n_test": 120, "epochs": 50, "n_steps": 5,
              "hidden": 48, "posterior_samples": 200, "seeds": [0],
              "n_calib_obs": 40, "n_mcmc_obs": 3, "mcmc_samples": 400},
    "full": {"n_train": 8000, "n_test": 1000, "epochs": 400, "n_steps": 20,
             "hidden": 128, "posterior_samples": 2000, "seeds": [0, 1, 2],
             "n_calib_obs": 200, "n_mcmc_obs": 20, "mcmc_samples": 2000},
}


def load_cfg(phase, mode="smoke"):
    cfg = dict(_DEFAULTS[mode])
    path = os.path.join(CONFIG_DIR, f"{phase}.yaml")
    if os.path.exists(path):
        try:
            import yaml
            with open(path) as f:
                y = yaml.safe_load(f) or {}
            cfg.update((y.get(mode) or {}))
        except Exception:
            pass
    cfg["mode"] = mode
    return cfg


# --------------------------------------------------------------- metrics
def relative_l2(true, pred):
    from prism.utils.metrics import relative_l2 as _r
    return float(_r(true, pred))


def rmse(true, pred):
    from prism.utils.metrics import rmse as _r
    return float(_r(true, pred))


def r2(true, pred):
    from prism.utils.metrics import r2_score as _r
    return float(_r(true, pred))


def violation_rate(X, constraint, meta=None):
    """Fraction of rows violating the task constraint (method-agnostic)."""
    X = np.asarray(X, float)
    if constraint == "positive":
        viol = (X <= 0).any(axis=1)
    elif constraint == "box":
        lo, hi = np.array(meta["box"][0]), np.array(meta["box"][1])
        viol = ((X < lo) | (X > hi)).any(axis=1)
    elif constraint == "simplex":
        viol = (np.abs(X.sum(1) - 1) > 1e-3) | (X < 0).any(axis=1)
    else:
        viol = np.zeros(len(X), bool)
    return float(viol.mean())


# --------------------------------------------------------------- methods
def make_prism(cfg, constraint="none", **over):
    from prism import PRISM
    kw = dict(epochs=cfg["epochs"], hidden=cfg["hidden"], solver="rk4",
              n_steps=cfg["n_steps"], constraint=constraint, seed=0)
    kw.update(over)
    return PRISM(**kw)


def make_method(name, cfg, constraint="none", **over):
    """Factory for any method by short name."""
    from prism import PRISM
    from baselines import CINN, NPECNF, AmortizedVI, FNO, DeepONet, PINN
    e, h = cfg["epochs"], cfg["hidden"]
    if name == "PRISM":
        return make_prism(cfg, constraint, **over)
    if name == "PRISM-soft":
        return PRISM(epochs=e, hidden=h, soft=True, constraint=constraint, seed=0, **over)
    if name == "PRISM-static":
        return make_prism(cfg, constraint, liquid=False, **over)
    if name == "cINN":
        return CINN(epochs=e, hidden=h, seed=0, **over)
    if name == "NPE-CNF":
        return NPECNF(epochs=e, hidden=h, solver="rk4", n_steps=cfg["n_steps"], seed=0, **over)
    if name == "VI":
        return AmortizedVI(epochs=e, hidden=h, seed=0, **over)
    if name == "FNO":
        return FNO(epochs=e, seed=0, **over)
    if name == "DeepONet":
        return DeepONet(epochs=e, seed=0, **over)
    if name == "PINN":
        return PINN(epochs=e, seed=0, **over)
    raise KeyError(name)


def fit_timed(model, *args):
    t0 = time.time()
    model.fit(*args)
    return model, time.time() - t0


# --------------------------------------------------------------- caching
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_results(phase, scalars, arrays=None):
    jpath = os.path.join(RESULTS_DIR, f"{phase}.json")
    with open(jpath, "w") as f:
        json.dump(scalars, f, indent=2, cls=NumpyEncoder)
    out = [jpath]
    if arrays:
        npath = os.path.join(RESULTS_DIR, f"{phase}.npz")
        np.savez_compressed(npath, **arrays)
        out.append(npath)
    return out


def load_results(phase):
    with open(os.path.join(RESULTS_DIR, f"{phase}.json")) as f:
        return json.load(f)


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")
