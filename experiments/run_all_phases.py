"""PRISM Phase 6 -- single consolidated run (the final results trial).

Executes every phase end-to-end at --mode full with 5 seeds into ONE manifest,
then regenerates every table and figure from that frozen manifest, then runs the
preflight audit on the assembled manuscript. This is the only run whose numbers
enter the camera-ready; because all tables/figures read from the same manifest,
no value can disagree across the paper.

Run in D:\\ICLR:  py experiments\\run_all_phases.py --mode full
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import importlib
import os
import shutil
import sys
import time

MANIFEST = "results/manifest.json"

# (module, human label). Each module exposes main(); we pass args via sys.argv.
EXPERIMENTS = [
    ("experiments.phase2_reversibility", "P2 reversibility"),
    ("experiments.phase2_constraints_fair", "P2 fair constraints"),
    ("experiments.phase2_stability", "P2 stability"),
    ("experiments.phase3_liquid_vs_static", "P3 liquid vs static"),
    ("experiments.phase3_conditioning", "P3 conditioning"),
    ("experiments.phase3_hybrid_velocity", "P3 hybrid velocity"),
    ("experiments.phase3_projection", "P3 projection"),
    ("experiments.phase4_calibration", "P4 calibration"),
    ("experiments.phase4_joint_geometry", "P4 joint geometry"),
    ("experiments.phase4_mixture_base", "P4 mixture base (optional)"),
    ("experiments.phase5_main_pde", "P5 main PDE"),
    ("experiments.phase5_efficiency_scaling", "P5 efficiency/scaling"),
    ("experiments.phase5_superres", "P5 super-resolution"),
    ("experiments.phase5_generality", "P5 generality"),
]

TABLES = ["tables.make_phase2_tables", "tables.make_phase3_tables",
          "tables.make_phase4_tables", "tables.make_phase5_tables"]
FIGURES = ["figures.make_phase2_figures", "figures.make_phase4_figures",
           "figures.make_phase5_figures"]


def _run(mod_name, mode, optional=False):
    argv = sys.argv
    sys.argv = [mod_name, "--mode", mode, "--manifest", MANIFEST]
    try:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        mod.main()
        return True
    except Exception as e:
        tag = "SKIP" if optional else "FAIL"
        print(f"  [{tag}] {mod_name}: {e}")
        return optional
    finally:
        sys.argv = argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full")
    ap.add_argument("--fresh", action="store_true", help="delete any existing manifest first")
    ap.add_argument("--manuscript", default="prism_iclr2027.tex")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    if args.fresh and os.path.exists(MANIFEST):
        shutil.move(MANIFEST, MANIFEST + f".bak.{int(time.time())}")
        print("archived previous manifest")

    print(f"\n=== consolidated run (mode={args.mode}) ===")
    t0 = time.time()
    for mod, label in EXPERIMENTS:
        print(f"-- {label}")
        _run(mod, args.mode, optional=("mixture" in mod))
    print(f"experiments done in {time.time()-t0:.0f}s")

    print("\n=== regenerate all tables/figures from the frozen manifest ===")
    for mod in TABLES + FIGURES:
        print(f"-- {mod}")
        _run(mod, args.mode)

    print("\n=== preflight audit ===")
    try:
        from tools.preflight_audit import audit_file, _print
        if os.path.exists(args.manuscript):
            ok = _print(audit_file(args.manuscript, root="."))
            print("\nFINAL:", "READY TO SUBMIT" if ok else "FIX ERRORS ABOVE")
        else:
            print(f"  (manuscript {args.manuscript} not found; skipping audit)")
    except Exception as e:
        print(f"  audit error: {e}")


if __name__ == "__main__":
    main()
