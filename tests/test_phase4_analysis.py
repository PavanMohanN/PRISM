"""Unit tests for Phase 4 analysis core (run without torch)."""
from __future__ import annotations

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase4_analysis import (
    sbc_ranks, sbc_uniformity, coverage_curve, ece_from_curve,
    mmd_rbf, mode_coverage, marginal_joint_gap,
)


def test_sbc_uniform_when_calibrated():
    rng = np.random.default_rng(0)
    N, L, d = 400, 100, 2
    # proper SBC: posterior N(mu,1); truth is an exchangeable draw from that same
    # posterior, so the rank of the truth among the samples is uniform.
    mu = rng.normal(size=(N, d))
    post = mu[:, None, :] + rng.normal(size=(N, L, d))
    truths = mu + rng.normal(size=(N, d))
    ranks = sbc_ranks(truths, post)
    stat, p = sbc_uniformity(ranks)
    assert p > 0.05, (stat, p)


def test_sbc_nonuniform_when_overconfident():
    rng = np.random.default_rng(1)
    N, L, d = 400, 100, 1
    truths = rng.normal(size=(N, d))
    # overconfident: posterior far too tight around a biased point -> truth in tails
    post = 0.05 * rng.normal(size=(N, L, d)) + 0.0
    ranks = sbc_ranks(truths, post)
    stat, p = sbc_uniformity(ranks)
    assert p < 0.01, (stat, p)


def test_coverage_and_ece():
    rng = np.random.default_rng(2)
    N, L, d = 500, 400, 2
    mu = rng.normal(size=(N, d))
    post = mu[:, None, :] + rng.normal(size=(N, L, d))     # posterior N(mu,1)
    truths = mu + rng.normal(size=(N, d))                  # exchangeable draw
    levels, emp = coverage_curve(truths, post, (0.5, 0.8, 0.9))
    assert ece_from_curve(levels, emp) < 0.05


def test_mmd_zero_same_dist_positive_diff():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 2)); Y = rng.normal(size=(300, 2))
    Z = rng.normal(size=(300, 2)) + 3.0
    assert mmd_rbf(X, Y) < 0.02
    assert mmd_rbf(X, Z) > 0.2


def test_mode_coverage_balanced_vs_collapsed():
    rng = np.random.default_rng(4)
    centers = np.array([[-2.0, 0.0], [2.0, 0.0]])
    balanced = np.vstack([rng.normal(centers[0], 0.3, (500, 2)),
                          rng.normal(centers[1], 0.3, (500, 2))])
    collapsed = rng.normal(centers[0], 0.3, (1000, 2))
    _, bal_balanced = mode_coverage(balanced, centers)
    _, bal_collapsed = mode_coverage(collapsed, centers)
    assert bal_balanced > 0.7
    assert bal_collapsed < 0.05


def test_marginal_ok_but_joint_wrong():
    # ref is correlated; post matches marginals (each N(0,1)) but not the joint.
    rng = np.random.default_rng(5)
    C = np.array([[1.0, 0.9], [0.9, 1.0]])
    ref = rng.multivariate_normal([0, 0], C, size=800)
    post = rng.normal(size=(800, 2))                   # independent -> right marginals
    marg, joint = marginal_joint_gap(post, ref)
    assert marg < 0.02
    assert joint > marg * 3        # joint discrepancy is much larger


if __name__ == "__main__":
    for fn in [test_sbc_uniform_when_calibrated, test_sbc_nonuniform_when_overconfident,
               test_coverage_and_ece, test_mmd_zero_same_dist_positive_diff,
               test_mode_coverage_balanced_vs_collapsed, test_marginal_ok_but_joint_wrong]:
        fn()
    rng = np.random.default_rng(5)
    C = np.array([[1.0, 0.9], [0.9, 1.0]])
    ref = rng.multivariate_normal([0, 0], C, size=800)
    post = rng.normal(size=(800, 2))
    marg, joint = marginal_joint_gap(post, ref)
    print(f"marginal MMD={marg:.4f}  joint MMD={joint:.4f}  (joint >> marginal => geometry wrong)")
    print("all phase-4 analysis tests: PASS")
