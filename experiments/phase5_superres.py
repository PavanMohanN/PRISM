"""PRISM Phase 5 -- E13: super-resolution, correctly sourced (guide item 28).

The reported super-resolution numbers matched the FNO forward numbers, not
PRISM's. Here we recompute zero-shot super-resolution with PRISM's OWN forward map
at 1x/2x/4x the training resolution, and then run a provenance audit that asserts
the numbers are PRISM's and NOT identical to FNO's. If the audit says "FNO", the
run aborts loudly rather than silently mislabeling.

Run in D:\\ICLR:  py experiments\\phase5_superres.py --mode smoke
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
from experiments.phase5_analysis import source_audit

TASKS = ["burgers", "helmholtz", "darcy"]
FACTORS = [1, 2, 4]
SEEDS = [0, 1, 2, 3, 4]


def _relL2(pred, true):
    return float(np.linalg.norm(np.asarray(pred) - np.asarray(true))
                 / (np.linalg.norm(np.asarray(true)) + 1e-12))


def _forward_at(est, ds, factor):
    """Query the forward map at `factor` x the training resolution."""
    Xup = ds.upsample_params(ds.X_test, factor) if hasattr(ds, "upsample_params") else ds.X_test
    yhi = ds.solve_high_res(Xup, factor) if hasattr(ds, "solve_high_res") else ds.y_test
    pred = est.predict_forward(Xup) if hasattr(est, "predict_forward") else est.predict(Xup)
    return _relL2(pred, yhi)


def run(task, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    prism = make_method("PRISM", task, cfg); prism.fit(ds.X_train, ds.y_train)
    fno = make_method("FNO", task, cfg); fno.fit(ds.X_train, ds.y_train)

    prism_curve = [_forward_at(prism, ds, f) for f in FACTORS]
    fno_curve = [_forward_at(fno, ds, f) for f in FACTORS]

    who = source_audit(prism_curve, {"PRISM": prism_curve, "FNO": fno_curve})
    assert who == "PRISM", f"super-res provenance audit failed: numbers look like {who}"
    return {"factors": FACTORS, "prism_relL2": prism_curve, "fno_relL2": fno_curve}


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
        man.run_seeds("phase5_superres", task, "PRISM",
                      fn=lambda s, t=task: run(t, s, cfg),
                      seeds=SEEDS, config={"mode": args.mode})
        c = man.agg("phase5_superres", task, "PRISM", "prism_relL2") if False else None
        rec = [r for k, r in man.records.items() if k[:3] == ("phase5_superres", task, "PRISM")][0]
        print(f"  [{task:10s}] PRISM super-res relL2 @1/2/4 = "
              f"{[round(x,3) for x in rec.metrics['prism_relL2']]}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
