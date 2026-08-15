"""Unit tests for Phase 3 analysis core (run without torch)."""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase3_analysis import (
    paired_delta, classify_regime, spread_ratio, y_dependence, param_parity,
)


def test_paired_delta_consistent_improvement():
    liquid = np.array([0.95, 0.96, 0.94, 0.955, 0.945])
    static = np.array([0.90, 0.91, 0.89, 0.905, 0.895])
    d = paired_delta(liquid, static)
    assert d["all_same_sign"] and d["mean"] > 0
    assert classify_regime(d, higher_is_better=True) == "helps"


def test_paired_delta_noise_is_neutral():
    # deltas of mixed sign -> not sign-consistent -> neutral regardless of effect
    a = np.array([0.91, 0.89, 0.92, 0.88, 0.90])
    b = np.array([0.90, 0.90, 0.90, 0.90, 0.90])
    d = paired_delta(a, b)
    assert not d["all_same_sign"]
    assert classify_regime(d, higher_is_better=True) == "neutral"


def test_regime_hurts():
    liquid = np.array([0.80, 0.81, 0.79])
    static = np.array([0.90, 0.91, 0.89])
    d = paired_delta(liquid, static)
    assert classify_regime(d, higher_is_better=True) == "hurts"


def test_spread_ratio_flags_collapse():
    rng = np.random.default_rng(1)
    prior_std = np.array([1.0, 1.0])
    informative = rng.normal(0, 0.2, size=(2000, 2))   # tight posterior
    collapsed = rng.normal(0, 1.0, size=(2000, 2))     # ~ prior width
    assert spread_ratio(informative, prior_std) < 0.5
    assert spread_ratio(collapsed, prior_std) > 0.8


def test_y_dependence_flags_collapse():
    prior_std = np.array([1.0, 1.0])
    # informative: posterior mean tracks y across observations
    pm_informative = np.stack([np.linspace(-1, 1, 20), np.linspace(1, -1, 20)], 1)
    # collapsed: posterior mean ~ constant regardless of y
    pm_collapsed = np.zeros((20, 2)) + 0.01 * np.random.default_rng(0).normal(size=(20, 2))
    assert y_dependence(pm_informative, prior_std) > 0.3
    assert y_dependence(pm_collapsed, prior_std) < 0.05


def test_param_parity():
    assert param_parity(10000, 10500)
    assert not param_parity(10000, 20000)


if __name__ == "__main__":
    for fn in [test_paired_delta_consistent_improvement, test_paired_delta_noise_is_neutral,
               test_regime_hurts, test_spread_ratio_flags_collapse,
               test_y_dependence_flags_collapse, test_param_parity]:
        fn()
    # illustrative
    liquid = np.array([0.95, 0.96, 0.94, 0.955, 0.945])
    static = np.array([0.90, 0.91, 0.89, 0.905, 0.895])
    d = paired_delta(liquid, static)
    print(f"liquid-vs-static delta: {d['mean']:+.3f} +/- {d['std']:.3f}, "
          f"effect={d['effect']:.1f}, verdict={classify_regime(d)}")
    print("all phase-3 analysis tests: PASS")
