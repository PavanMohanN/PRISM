"""Unit tests for Phase 5 analysis core (run without torch)."""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase5_analysis import (
    crossover_observations, total_costs, source_audit, relative_gap, primary_metric,
)


def test_crossover_exists():
    # train 100s; amortized 0.01s/sample x 100 samples = 1.0s/obs;
    # reference 2.5s/obs -> crossover at 100/(2.5-1.0) = 66.7 observations
    m = crossover_observations(100.0, 0.01, 100, 2.5)
    assert abs(m - 66.666) < 1e-2, m
    amort, ref = total_costs([0, 67, 200], 100.0, 0.01, 100, 2.5)
    assert amort[0] > ref[0]        # amortized more expensive at few observations
    assert amort[-1] < ref[-1]      # cheaper at many observations


def test_crossover_never():
    # reference cheaper per obs than amortized per obs -> never wins
    m = crossover_observations(100.0, 0.05, 100, 2.5)  # 5.0/obs > 2.5/obs
    assert m == float("inf")


def test_crossover_shrinks_with_dimension():
    # as dimension grows, MCMC per-target cost grows -> crossover happens sooner
    per_target = {2: 1.5, 8: 4.0, 32: 20.0}
    ms = [crossover_observations(100.0, 0.01, 100, per_target[d]) for d in (2, 8, 32)]
    assert ms[0] > ms[1] > ms[2]


def test_source_audit_catches_wrong_method():
    fno = np.array([0.092, 0.776, 0.255])
    prism = np.array([0.069, 0.429, 0.294])
    reported = np.array([0.092, 0.776, 0.255])   # these are actually FNO's!
    who = source_audit(reported, {"PRISM": prism, "FNO": fno})
    assert who == "FNO"                          # audit flags the mislabel
    who2 = source_audit(prism, {"PRISM": prism, "FNO": fno})
    assert who2 == "PRISM"


def test_relative_gap():
    # PRISM forward 0.069 vs best (PINN) 0.020 -> ~245% worse
    g = relative_gap(0.069, 0.020, lower_is_better=True)
    assert 2.4 < g < 2.5


def test_primary_metric():
    assert primary_metric("darcy") == "inverse_r2"
    assert primary_metric("two_moons") == "c2st"


if __name__ == "__main__":
    for fn in [test_crossover_exists, test_crossover_never,
               test_crossover_shrinks_with_dimension,
               test_source_audit_catches_wrong_method, test_relative_gap,
               test_primary_metric]:
        fn()
    m = crossover_observations(100.0, 0.01, 100, 2.5)
    print(f"amortization break-even at M = {m:.1f} observations")
    ms = [crossover_observations(100.0, 0.01, 100, pt) for pt in (1.5, 4.0, 20.0)]
    print(f"crossover vs dim (per-target 1.5/4/20): {[round(x,1) for x in ms]}")
    print("all phase-5 analysis tests: PASS")
