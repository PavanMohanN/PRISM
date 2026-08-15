"""PRISM Phase 1 -- NPE-CNF re-validation (guide item 12).

The reported NPE-CNF was suspiciously weak on Gaussian-Linear (C2ST ~0.94, 100%
coverage) -- a sign of under-training / poor conditioning rather than a property
of the method. This module provides:

  1. RECOMMENDED_NPE_CNF: a stronger, documented training config.
  2. gaussian_linear_reference(): the ANALYTIC posterior of a linear-Gaussian
     model, so any posterior estimator can be sanity-checked against ground truth.
  3. sanity_check(): asserts a posterior estimator recovers the near-Gaussian
     conditional (C2ST -> ~0.5, coverage -> nominal). Use this as a GATE before
     trusting any NPE-CNF numbers in the paper.

The sanity check is method-agnostic: it takes any object exposing
`predict_posterior(y, n)` returning samples in the SAME coordinate as the truth.
"""
from __future__ import annotations

import numpy as np

# ---- stronger, explicit training configuration (document this in the appendix) ----
RECOMMENDED_NPE_CNF = dict(
    hidden_width=128,
    n_layers=5,
    activation="silu",
    epochs=400,           # was far too low in the flagged run
    batch_size=256,
    lr=3e-4,
    lr_schedule="cosine",
    weight_decay=1e-5,
    standardize_inputs=True,     # critical for conditioning
    standardize_targets=True,
    grad_clip=5.0,
    early_stop_patience=40,
    n_seeds=5,
    backbone="neural_spline_flow",   # trusted SBI stack default (Durkan et al. 2019)
)


# ---------------------------------------------------------------------------
# Analytic linear-Gaussian model:  y = A x + b + noise,  x ~ N(0, S0),
# noise ~ N(0, Sn).  Posterior is Gaussian with closed form.
# ---------------------------------------------------------------------------
def gaussian_linear_reference(A, b, S0, Sn, y):
    A = np.asarray(A, float); b = np.asarray(b, float)
    S0 = np.asarray(S0, float); Sn = np.asarray(Sn, float); y = np.asarray(y, float)
    Sn_inv = np.linalg.inv(Sn); S0_inv = np.linalg.inv(S0)
    prec = S0_inv + A.T @ Sn_inv @ A
    cov = np.linalg.inv(prec)
    mean = cov @ (A.T @ Sn_inv @ (y - b))
    return mean, cov


def _c2st(xs, ys, seed=0):
    """Classifier two-sample test accuracy (chance = 0.5). Needs scikit-learn."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(seed)
    X = np.vstack([xs, ys]); lab = np.r_[np.zeros(len(xs)), np.ones(len(ys))]
    idx = rng.permutation(len(lab)); X, lab = X[idx], lab[idx]
    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=300, random_state=seed)
    return float(np.mean(cross_val_score(clf, X, lab, cv=5)))


def coverage_at(levels, ref_mean, ref_cov, samples):
    """Empirical central-interval coverage of `samples` vs the analytic Gaussian."""
    from scipy.stats import chi2
    d = ref_mean.shape[0]
    cov_inv = np.linalg.inv(ref_cov)
    diff = samples - ref_mean
    md2 = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)  # squared Mahalanobis
    out = {}
    for lv in levels:
        thresh = chi2.ppf(lv, df=d)
        out[lv] = float(np.mean(md2 <= thresh))
    return out


def sanity_check(estimator, A, b, S0, Sn, y_obs, n_samples=2000,
                 c2st_max=0.65, cov_tol=0.1, levels=(0.5, 0.8, 0.9)):
    """GATE: return (passed, report) for a posterior estimator on Gaussian-Linear.

    Passes iff C2ST-to-analytic <= c2st_max AND |coverage - nominal| <= cov_tol
    at every level. A strong NPE-CNF MUST pass this trivial task.
    """
    ref_mean, ref_cov = gaussian_linear_reference(A, b, S0, Sn, y_obs)
    ref_samples = np.random.default_rng(0).multivariate_normal(
        ref_mean, ref_cov, size=n_samples)
    est_samples = np.asarray(estimator.predict_posterior(y_obs, n_samples))
    est_samples = est_samples.reshape(-1, ref_mean.shape[0])

    c2st = _c2st(est_samples, ref_samples)
    cov = coverage_at(levels, ref_mean, ref_cov, est_samples)
    cov_ok = all(abs(cov[lv] - lv) <= cov_tol for lv in levels)
    passed = (c2st <= c2st_max) and cov_ok
    report = {"c2st_to_analytic": c2st, "coverage": cov,
              "c2st_max": c2st_max, "cov_tol": cov_tol, "passed": passed}
    return passed, report


if __name__ == "__main__":
    # Self-demo: the ANALYTIC posterior must pass its own sanity check.
    rng = np.random.default_rng(0)
    d, m = 3, 4
    A = rng.normal(size=(m, d)); b = rng.normal(size=m)
    S0 = np.eye(d); Sn = 0.2 * np.eye(m)
    x_true = rng.normal(size=d); y = A @ x_true + b + rng.multivariate_normal(np.zeros(m), Sn)

    class _Analytic:
        def predict_posterior(self, y, n):
            mean, cov = gaussian_linear_reference(A, b, S0, Sn, y)
            return rng.multivariate_normal(mean, cov, size=n)

    ok, rep = sanity_check(_Analytic(), A, b, S0, Sn, y)
    print("analytic self-check passed:", ok)
    print("c2st_to_analytic:", round(rep["c2st_to_analytic"], 3), "coverage:", rep["coverage"])
