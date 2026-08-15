"""Posterior calibration metrics.

These quantify how trustworthy a posterior is, independent of point accuracy:

* ``c2st``            classifier 2-sample test vs a reference (0.5 = indistinguishable)
* ``coverage``       empirical coverage of central credible intervals
* ``sbc_ranks``      simulation-based calibration rank statistics (uniform if calibrated)
* ``crps``           continuous ranked probability score (lower better)
* ``gaussian_nll``   Gaussian-approx negative log-prob of the truth (lower better)
* ``wasserstein_to_reference``  per-dim W1 distance to reference samples

All operate on posterior *samples* so they apply to any method (PRISM, flows,
VI, MCMC) uniformly.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------- C2ST
def c2st(samples_p, samples_q, n_folds=5, seed=0):
    """Classifier two-sample test accuracy (0.5 = indistinguishable).

    Trains a small classifier to tell reference samples (p) from model samples
    (q) under cross-validation; returns mean held-out accuracy.
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    p = np.asarray(samples_p, float)
    q = np.asarray(samples_q, float)
    n = min(len(p), len(q))
    p, q = p[:n], q[:n]
    X = np.vstack([p, q])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X = StandardScaler().fit_transform(X)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300,
                            early_stopping=True, random_state=seed)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


# ----------------------------------------------------------------- coverage
def coverage(thetas_true, posterior_samples, levels=(0.5, 0.8, 0.9, 0.95)):
    """Empirical central-interval coverage.

    thetas_true:        (M, d) true parameters
    posterior_samples:  (M, S, d) samples per observation
    Returns dict with per-level empirical coverage (averaged over dims) and a
    scalar expected-coverage-error |empirical - nominal|.
    """
    thetas_true = np.asarray(thetas_true, float)
    S = posterior_samples
    M, _, d = S.shape
    out = {}
    err = 0.0
    for lvl in levels:
        lo_q = (1 - lvl) / 2
        hi_q = 1 - lo_q
        lo = np.quantile(S, lo_q, axis=1)          # (M, d)
        hi = np.quantile(S, hi_q, axis=1)
        inside = (thetas_true >= lo) & (thetas_true <= hi)
        emp = inside.mean()
        out[lvl] = float(emp)
        err += abs(emp - lvl)
    out["expected_coverage_error"] = float(err / len(levels))
    return out


# ---------------------------------------------------------------------- SBC
def sbc_ranks(thetas_true, posterior_samples):
    """Per-dim SBC ranks of the truth within posterior samples.

    Returns ranks (M, d) in [0, S]; uniform histogram <=> calibrated. Also
    returns a KS statistic vs uniform (lower better).
    """
    thetas_true = np.asarray(thetas_true, float)
    S = posterior_samples
    M, n, d = S.shape
    ranks = (S < thetas_true[:, None, :]).sum(axis=1)     # (M, d)
    # KS vs discrete uniform on {0..n}
    ks = []
    for j in range(d):
        r = np.sort(ranks[:, j]) / n
        cdf = np.arange(1, M + 1) / M
        ks.append(np.max(np.abs(r - cdf)))
    return ranks, float(np.mean(ks))


# --------------------------------------------------------------------- CRPS
def crps(thetas_true, posterior_samples):
    """Mean per-dim empirical CRPS (lower better)."""
    thetas_true = np.asarray(thetas_true, float)
    S = posterior_samples
    M, n, d = S.shape
    term1 = np.abs(S - thetas_true[:, None, :]).mean(axis=1)        # (M,d)
    # E|X - X'| via a subsample to stay O(n)
    idx = np.random.default_rng(0).permutation(n)
    term2 = np.abs(S - S[:, idx, :]).mean(axis=1)                   # (M,d)
    val = (term1 - 0.5 * term2).mean()
    return float(val)


# ------------------------------------------------------------- Gaussian NLL
def gaussian_nll(thetas_true, posterior_samples, eps=1e-6):
    """Negative log-prob of truth under a Gaussian fit to the samples."""
    thetas_true = np.asarray(thetas_true, float)
    S = posterior_samples
    mu = S.mean(axis=1)                                    # (M,d)
    var = S.var(axis=1) + eps
    nll = 0.5 * (((thetas_true - mu) ** 2) / var + np.log(2 * np.pi * var))
    return float(nll.sum(axis=1).mean())


# ---------------------------------------------------- Wasserstein-to-reference
def wasserstein_to_reference(samples, reference):
    """Mean per-dimension 1-Wasserstein distance to reference samples."""
    from scipy.stats import wasserstein_distance
    samples = np.asarray(samples, float)
    reference = np.asarray(reference, float)
    d = samples.shape[1]
    return float(np.mean([wasserstein_distance(samples[:, j], reference[:, j])
                          for j in range(d)]))


def summarize(thetas_true, posterior_samples, reference=None):
    """Convenience bundle of sample-based metrics for a batch of observations."""
    res = {}
    cov = coverage(thetas_true, posterior_samples)
    res["expected_coverage_error"] = cov["expected_coverage_error"]
    res["coverage"] = {k: v for k, v in cov.items() if k != "expected_coverage_error"}
    _, ks = sbc_ranks(thetas_true, posterior_samples)
    res["sbc_ks"] = ks
    res["crps"] = crps(thetas_true, posterior_samples)
    res["gaussian_nll"] = gaussian_nll(thetas_true, posterior_samples)
    return res
