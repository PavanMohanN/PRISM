"""PRISM Phase 4 -- analysis core (no torch; unit-tested).

Posterior-quality diagnostics that back the Phase 4 tables and figures:

  * sbc_ranks / sbc_uniformity -- simulation-based calibration across ALL domains
    (item 17): calibrated posteriors give uniform ranks.
  * coverage_curve / ece_from_curve -- empirical vs nominal coverage.
  * mmd_rbf -- kernel two-sample discrepancy (a joint-distribution metric).
  * mode_coverage -- do posterior samples cover BOTH reference modes? (item 19)
  * marginal_joint_gap -- marginal calibration can look fine while the JOINT
    geometry is wrong; this returns both so the paper can show the gap (item 19).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chisquare


# ----------------------------- SBC -----------------------------
def sbc_ranks(truths, post_samples):
    """Rank of each true parameter among its posterior samples (per dim).

    truths: [N, d].  post_samples: [N, L, d].  Returns ranks [N, d] in {0..L}.
    """
    truths = np.asarray(truths, float)
    post = np.asarray(post_samples, float)
    return (post < truths[:, None, :]).sum(axis=1)      # [N, d]


def sbc_uniformity(ranks, n_bins=10):
    """Chi-square uniformity test on SBC ranks. Returns (stat, p_value).

    High p => ranks consistent with uniform => calibrated.
    """
    ranks = np.asarray(ranks).ravel().astype(float)
    L = ranks.max() if ranks.max() > 0 else 1.0
    hist, _ = np.histogram(ranks / L, bins=n_bins, range=(0, 1))
    expected = np.full(n_bins, hist.sum() / n_bins)
    stat, p = chisquare(hist, expected)
    return float(stat), float(p)


# --------------------------- coverage --------------------------
def coverage_curve(truths, post_samples, levels=(0.5, 0.8, 0.9, 0.95)):
    """Empirical central-interval coverage (averaged over dims and observations)."""
    truths = np.asarray(truths, float)
    post = np.asarray(post_samples, float)                # [N, L, d]
    emp = []
    for a in levels:
        lo = np.quantile(post, (1 - a) / 2, axis=1)       # [N, d]
        hi = np.quantile(post, (1 + a) / 2, axis=1)
        inside = (truths >= lo) & (truths <= hi)
        emp.append(float(inside.mean()))
    return list(levels), emp


def ece_from_curve(levels, emp):
    """Expected coverage error: mean |empirical - nominal|."""
    return float(np.mean(np.abs(np.asarray(emp) - np.asarray(levels))))


# ----------------------------- MMD -----------------------------
def _median_bw(X, Y):
    Z = np.vstack([X, Y])
    n = min(len(Z), 200)
    idx = np.random.default_rng(0).choice(len(Z), n, replace=False)
    d = np.sqrt(((Z[idx, None, :] - Z[None, idx, :]) ** 2).sum(-1))
    med = np.median(d[d > 0])
    return med if med > 0 else 1.0


def mmd_rbf(X, Y, bandwidth=None):
    """Unbiased RBF-kernel MMD^2 between two sample sets."""
    X = np.atleast_2d(np.asarray(X, float)); Y = np.atleast_2d(np.asarray(Y, float))
    if X.shape[0] == 1 and X.shape[1] != Y.shape[1]:
        X = X.T
    bw = bandwidth or _median_bw(X, Y)
    g = 1.0 / (2 * bw ** 2)

    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-g * d2)

    m, n = len(X), len(Y)
    Kxx = k(X, X); Kyy = k(Y, Y); Kxy = k(X, Y)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return float(Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1))
                - 2 * Kxy.mean())


# ------------------------- mode coverage -----------------------
def mode_coverage(samples, centers):
    """Fraction of samples nearest each reference mode center.

    Returns proportions [K] and a balance score in [0,1] (1 = perfectly balanced).
    A unimodal-base flow that covers only one mode gives an imbalanced split.
    """
    samples = np.asarray(samples, float); centers = np.asarray(centers, float)
    d2 = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    assign = d2.argmin(1)
    props = np.array([(assign == k).mean() for k in range(len(centers))])
    balance = float(props.min() / (props.max() + 1e-12))
    return props, balance


def marginal_joint_gap(post, ref):
    """(mean 1-D marginal MMD, joint MMD). A large joint with small marginal gap
    is the signature of correct marginals but wrong joint geometry (item 19)."""
    post = np.asarray(post, float); ref = np.asarray(ref, float)
    d = post.shape[1]
    marg = np.mean([mmd_rbf(post[:, i:i + 1], ref[:, i:i + 1]) for i in range(d)])
    joint = mmd_rbf(post, ref)
    return float(marg), float(joint)
