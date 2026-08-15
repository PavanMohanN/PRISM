"""Calibration metric sanity checks."""
import numpy as np
from prism.posterior import calibration as cal


def test_c2st_identical_near_half():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((400, 3)); b = rng.standard_normal((400, 3))
    assert cal.c2st(a, b, seed=0) < 0.65


def test_c2st_shifted_is_high():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((400, 3)); b = rng.standard_normal((400, 3)) + 3.0
    assert cal.c2st(a, b, seed=0) > 0.8


def test_coverage_well_calibrated():
    # calibrated: truth and posterior samples are independent draws from N(0,1),
    # so central intervals cover truth at ~nominal rate
    rng = np.random.default_rng(0)
    M, S, d = 400, 800, 2
    truth = rng.standard_normal((M, d))
    samples = rng.standard_normal((M, S, d))
    cov = cal.coverage(truth, samples)
    assert cov["expected_coverage_error"] < 0.05


def test_sbc_crps_nll_finite():
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((100, 2))
    samples = truth[:, None, :] + rng.standard_normal((100, 200, 2))
    _, ks = cal.sbc_ranks(truth, samples)
    assert 0 <= ks <= 1
    assert np.isfinite(cal.crps(truth, samples))
    assert np.isfinite(cal.gaussian_nll(truth, samples))


def test_wasserstein_identical_is_small():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2000, 3))
    assert cal.wasserstein_to_reference(a, a) < 0.05
