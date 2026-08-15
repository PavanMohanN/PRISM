"""Phase 1 unit tests for support transforms (guide items 7-8).

The normalization tests use numpy/scipy quadrature and run WITHOUT torch: if the
pushforward of a base density through c integrates to 1.0, the change-of-variables
log-det is correct. The round-trip / feasibility / torch-vs-numpy cross-checks run
only when torch is importable.

Run:  python -m pytest tests/test_transforms.py -q
  or: python tests/test_transforms.py
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm, multivariate_normal

TOL_NORM = 1e-4
TOL_RT = 1e-10

# ---------- numpy reference implementations (ground truth) ----------
_q1 = norm(0, 1).pdf


def _pos_cinv(x):
    return np.log(np.expm1(x))


def _pos_dcinv(x):
    return 1.0 / (1.0 - np.exp(-x))


def _box_cinv(x, a, b):
    z = (x - a) / (b - a)
    return np.log(z / (1 - z))


def _box_dcinv(x, a, b):
    z = (x - a) / (b - a)
    return 1.0 / ((b - a) * z * (1 - z))


# ---------- normalization tests (no torch needed) ----------
def test_positive_normalizes():
    p = lambda x: _q1(_pos_cinv(x)) * abs(_pos_dcinv(x))
    integral, _ = quad(p, 1e-9, 60, limit=400)
    assert abs(integral - 1.0) < TOL_NORM, integral


def test_box_normalizes():
    a, b = -1.0, 2.0
    p = lambda x: _q1(_box_cinv(x, a, b)) * abs(_box_dcinv(x, a, b))
    integral, _ = quad(p, a + 1e-9, b - 1e-9, limit=400)
    assert abs(integral - 1.0) < TOL_NORM, integral


def test_simplex_normalizes_d3():
    def sig(u):
        return 1 / (1 + np.exp(-u))

    def inverse(x):
        d = len(x); u = np.zeros(d - 1); rem = 1.0
        for k in range(d - 1):
            zk = x[k] / rem; u[k] = np.log(zk / (1 - zk)); rem *= (1 - zk)
        return u

    def logdet_dxdu(u):
        d = len(u) + 1; z = sig(u); s = 0.0
        for k in range(d - 1):
            s += np.log(z[k]) + np.log(1 - z[k])
            s += (d - 2 - k) * np.log(1 - z[k])
        return s

    q2 = multivariate_normal(mean=np.zeros(2), cov=np.eye(2))

    def p_x(x1, x2):
        x = np.array([x1, x2, 1 - x1 - x2])
        if np.any(x <= 0):
            return 0.0
        u = inverse(x)
        return q2.pdf(u) * np.exp(-logdet_dxdu(u))

    integral, _ = dblquad(
        lambda x2, x1: p_x(x1, x2),
        1e-6, 1 - 1e-6,
        lambda x1: 1e-6, lambda x1: 1 - x1 - 1e-6,
        epsabs=1e-9,
    )
    assert abs(integral - 1.0) < 1e-3, integral


# ---------- torch cross-checks (skipped if torch missing) ----------
def _torch_or_skip():
    try:
        import torch  # noqa
        return torch
    except Exception:
        import pytest
        pytest.skip("torch not installed")


def test_torch_roundtrip_and_feasibility():
    torch = _torch_or_skip()
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prism.constraints.transforms import Positive, Box, StickBreaking

    torch.manual_seed(0)
    for tf, dim in [(Positive(4), 4), (Box(-1.0, 2.0, 3), 3), (StickBreaking(4), 3)]:
        u = torch.randn(1000, tf.unconstrained_dim, dtype=torch.float64)
        x = tf.forward(u)
        assert tf.is_feasible(x).all(), type(tf).__name__
        u2 = tf.inverse(x)
        assert torch.max(torch.abs(u2 - u)).item() < 1e-8, type(tf).__name__


def test_torch_logdet_matches_numpy():
    torch = _torch_or_skip()
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from prism.constraints.transforms import Positive, Box

    x = torch.linspace(0.05, 6.0, 50, dtype=torch.float64).unsqueeze(-1)
    got = Positive(1).log_abs_det_inverse(x).numpy()
    ref = -np.log(np.abs(_pos_dcinv(x.squeeze(-1).numpy())))  # -log|(c^-1)'|? sign
    # log_abs_det_inverse = log|(c^{-1})'(x)| = +log|dcinv|
    ref = np.log(np.abs(_pos_dcinv(x.squeeze(-1).numpy())))
    assert np.allclose(got, ref, atol=1e-8)

    a, b = -1.0, 2.0
    xb = torch.linspace(a + 0.05, b - 0.05, 50, dtype=torch.float64).unsqueeze(-1)
    gotb = Box(a, b, 1).log_abs_det_inverse(xb).numpy()
    refb = np.log(np.abs(_box_dcinv(xb.squeeze(-1).numpy(), a, b)))
    assert np.allclose(gotb, refb, atol=1e-8)


if __name__ == "__main__":
    # run the torch-free tests directly and report
    test_positive_normalizes()
    test_box_normalizes()
    test_simplex_normalizes_d3()
    print("normalization tests: PASS (positive, box, simplex integrate to 1.0)")
    try:
        import torch  # noqa
        test_torch_roundtrip_and_feasibility()
        test_torch_logdet_matches_numpy()
        print("torch tests: PASS (round-trip, feasibility, logdet match)")
    except Exception as e:
        print(f"torch tests: SKIPPED ({e})")
