"""Run every experiment phase end-to-end and cache results to results/.

    python experiments/run_all.py --smoke     # minutes on CPU, proves the pipeline
    python experiments/run_all.py --full       # paper-scale (GPU, multi-seed)

Each phase writes results/<phase>.json (+ .npz). figures/make_figures.py and
tables/make_tables.py then read only results/ to regenerate paper_assets/.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time
import traceback

import phase2_pde_core
import phase3_operator_scaling
import phase4_calibration
import phase5_generality
import phase6_ablation_robust_eff
from _common import banner

PHASES = [
    ("phase2", phase2_pde_core.run),
    ("phase3", phase3_operator_scaling.run),
    ("phase4", phase4_calibration.run),
    ("phase5", phase5_generality.run),
    ("phase6", phase6_ablation_robust_eff.run),
]


def main(mode):
    banner(f"PRISM experiment suite — mode={mode}")
    t0 = time.time()
    status = {}
    for name, fn in PHASES:
        try:
            ts = time.time()
            fn(mode)
            status[name] = f"ok ({time.time()-ts:.1f}s)"
        except Exception:
            status[name] = "FAILED"
            print(f"\n!!! {name} failed:\n{traceback.format_exc()}")
    banner("SUMMARY")
    for k, v in status.items():
        print(f"  {k:8s} {v}")
    print(f"\ntotal {time.time()-t0:.1f}s")
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    main("full" if a.full else "smoke")
