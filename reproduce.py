#!/usr/bin/env python3
"""One-command reproduction of every PRISM paper asset.

    python reproduce.py --smoke         # fast CPU sanity (minutes): experiments + assets
    python reproduce.py --full          # paper-scale (GPU, multi-seed): experiments + assets
    python reproduce.py --assets-only   # skip experiments, rebuild tables+figures from results/

Stages run as independent subprocesses so each gets a clean import environment:
    1. experiments/run_all.py   -> results/<phase>.json|npz
    2. tables/make_tables.py    -> paper_assets/T0..T8.tex
    3. figures/make_figures.py  -> paper_assets/F1..F13.pdf
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def _run(script, *script_args):
    cmd = [sys.executable, os.path.join(ROOT, script), *script_args]
    print(f"\n$ {' '.join(os.path.relpath(c, ROOT) if c.startswith(ROOT) else c for c in cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="fast CPU run")
    ap.add_argument("--full", action="store_true", help="paper-scale run")
    ap.add_argument("--assets-only", action="store_true",
                    help="rebuild tables/figures from cached results/")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    a = ap.parse_args()

    t0 = time.time()
    rc = 0
    if not a.assets_only:
        mode = "--full" if a.full else "--smoke"
        rc = _run("experiments/run_all.py", mode)
        if rc != 0:
            print("\n[reproduce] experiments reported failures; building assets from "
                  "whatever cached anyway.")
    rc |= _run("tables/make_tables.py")
    if not a.no_figures:
        rc |= _run("figures/make_figures.py")

    print(f"\n{'='*60}\n[reproduce] done in {time.time()-t0:.1f}s — assets in paper_assets/\n{'='*60}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
