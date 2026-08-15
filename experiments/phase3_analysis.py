"""PRISM Phase 3 -- analysis core (no torch; unit-tested).

Turns paired ablation measurements into defensible verdicts:

  * paired_delta / classify_regime -- for "does the liquid field help?" (item 15)
    and "does the hybrid velocity help?" (item 14). Seeds are PAIRED (same seed
    for both variants), so we report the per-seed difference, its consistency of
    sign, and an effect size relative to seed variance -- not a single-run gap.

  * spread_ratio / y_dependence -- collapse diagnostics for the conditioning
    ablation (item 13). A variant that "collapses to the prior" has posterior
    spread ~ prior spread AND a posterior summary that barely moves with y.

  * param_parity -- guards the matched-parameter-count requirement (item 15): a
    liquid-vs-static comparison is only fair if the two have ~equal capacity.
"""
from __future__ import annotations

import numpy as np


def paired_delta(a, b):
    """Per-seed difference delta = a - b (a, b aligned by seed).

    Returns mean, std, n, whether every seed agrees in sign, and an effect size
    |mean| / (std + eps).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1
    d = a - b
    mean = float(d.mean())
    std = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    same_sign = bool(np.all(d > 0) or np.all(d < 0))
    effect = float(abs(mean) / (std + 1e-12))
    return {"mean": mean, "std": std, "n": len(d),
            "all_same_sign": same_sign, "effect": effect}


def classify_regime(delta, higher_is_better=True, effect_thresh=1.0):
    """Verdict for a paired_delta result: 'helps' / 'neutral' / 'hurts'.

    Requires sign-consistency across seeds AND an effect size above threshold to
    call anything other than 'neutral'. This prevents seed noise from being read
    as a real effect (item 16).
    """
    if not delta["all_same_sign"] or delta["effect"] < effect_thresh:
        return "neutral"
    improves = (delta["mean"] > 0) == higher_is_better
    return "helps" if improves else "hurts"


def spread_ratio(post_samples, prior_std):
    """Mean over dims of std(posterior)/std(prior). ~1 (or huge) => collapse."""
    post_samples = np.asarray(post_samples, float).reshape(-1, np.shape(post_samples)[-1])
    ps = post_samples.std(0)
    pr = np.asarray(prior_std, float)
    return float(np.mean(ps / (pr + 1e-12)))


def y_dependence(posterior_means, prior_std):
    """How much the posterior MEAN moves as the observation y varies.

    posterior_means: [n_obs, dim] posterior means for different observations.
    Returns normalized dispersion; ~0 => posterior ignores y (collapsed).
    """
    pm = np.asarray(posterior_means, float)
    disp = pm.std(0)                       # variation across observations, per dim
    pr = np.asarray(prior_std, float)
    return float(np.mean(disp / (pr + 1e-12)))


def param_parity(params_a, params_b, tol=0.10):
    """True if two variants have parameter counts within `tol` relative diff."""
    hi = max(params_a, params_b); lo = min(params_a, params_b)
    return (hi - lo) / hi <= tol


def latex_pm(mean, std, prec=3):
    return f"{mean:.{prec}f} $\\pm$ {std:.{prec}f}"
