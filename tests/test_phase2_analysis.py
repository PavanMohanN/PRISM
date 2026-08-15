"""Unit tests for Phase 2 analysis core (run without torch)."""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase2_analysis import (
    loglog_order_and_floor, spectral_norm, lipschitz_bound, local_sensitivity_ratio,
)


def test_order_recovery_rk4_like():
    # error = C h^4 + floor, floor = 1e-12 (float64-like). Broad step range.
    steps = np.array([2, 4, 8, 16, 32, 64, 128, 256], float)
    h = 1.0 / steps
    errors = 0.5 * h ** 4 + 1e-12
    out = loglog_order_and_floor(steps, errors)
    assert abs(out["order"] - 4.0) < 0.2, out["order"]
    assert out["floor"] < 1e-9, out["floor"]


def test_order_recovery_rk2_like():
    steps = np.array([2, 4, 8, 16, 32, 64, 128, 256], float)
    h = 1.0 / steps
    errors = 0.3 * h ** 2 + 1e-13
    out = loglog_order_and_floor(steps, errors)
    assert abs(out["order"] - 2.0) < 0.2, out["order"]


def test_floor_detected_when_all_flat():
    # the ORIGINAL failure mode: everything already at the roundoff floor.
    steps = np.array([2, 4, 8, 16, 32], float)
    errors = np.full_like(steps, 4.4e-5)
    out = loglog_order_and_floor(steps, errors)
    # essentially no descending regime -> few/no fit points, floor ~ 4.4e-5
    assert abs(out["floor"] - 4.4e-5) < 1e-6
    assert out["n_fit"] <= 1


def test_spectral_norm_matches_numpy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        A = rng.normal(size=(7, 5))
        got = spectral_norm(A)
        ref = np.linalg.norm(A, 2)
        assert abs(got - ref) < 1e-4 * max(1.0, ref), (got, ref)


def test_bound_composition():
    assert abs(lipschitz_bound(2.0, 0.0, 3.0) - 6.0) < 1e-12
    assert lipschitz_bound(1.0, 1.0, 1.0) > np.e - 1e-9


def test_local_sensitivity_linear_map():
    # h(y) = M y  -> local ratio should approach ||M||_2 for the max direction
    rng = np.random.default_rng(2)
    M = rng.normal(size=(3, 3))
    mean_r, max_r = local_sensitivity_ratio(lambda y: M @ y, np.zeros(3),
                                             eps=1e-3, n_dirs=200)
    assert max_r <= np.linalg.norm(M, 2) + 1e-6
    assert mean_r <= max_r + 1e-9


if __name__ == "__main__":
    test_order_recovery_rk4_like()
    test_order_recovery_rk2_like()
    test_floor_detected_when_all_flat()
    test_spectral_norm_matches_numpy()
    test_bound_composition()
    test_local_sensitivity_linear_map()
    # print a couple of illustrative numbers
    steps = np.array([2, 4, 8, 16, 32, 64, 128, 256], float)
    errors = 0.5 * (1.0 / steps) ** 4 + 1e-12
    o = loglog_order_and_floor(steps, errors)
    print(f"RK4-like sweep: estimated order = {o['order']:.3f}, floor = {o['floor']:.1e}, "
          f"fit points = {o['n_fit']}")
    print("all phase-2 analysis tests: PASS")
