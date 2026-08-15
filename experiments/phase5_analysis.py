"""PRISM Phase 5 -- analysis core (no torch; unit-tested).

Backs the accuracy / efficiency / scaling / generality tables and figures:

  * crossover_observations -- turns the unsupported "amortization is faster" claim
    into a demonstrated break-even point: amortized inference pays a fixed training
    cost, then cheap per-sample cost; a per-observation reference (MCMC) pays no
    training but an expensive per-target cost. We report the number of test
    observations beyond which amortization wins in TOTAL wall-clock (item 20).

  * source_audit -- guards against the Table 8 / Figure 8 provenance bug (item 28):
    given a set of reported numbers and the candidate methods' known numbers, it
    reports which method the numbers actually came from.

  * relative_gap -- honest forward-accuracy wording (item 29): the numeric gap
    between PRISM and the best dedicated operator, instead of vague "competitive."

  * primary_metric -- posterior-appropriate metric per task so the generality
    table never scores a multimodal posterior by point R^2 (item 18).
"""
from __future__ import annotations

import numpy as np


def crossover_observations(train_cost, per_sample_amortized, n_samples,
                           per_target_reference):
    """Number of observations M* beyond which amortized total cost < reference.

    amortized_total(M) = train_cost + M * n_samples * per_sample_amortized
    reference_total(M) = M * per_target_reference
    Returns M* >= 0, or float('inf') if amortization never wins.
    """
    per_obs_amort = n_samples * per_sample_amortized
    denom = per_target_reference - per_obs_amort
    if denom <= 0:
        return float("inf")
    return max(0.0, train_cost / denom)


def total_costs(M_values, train_cost, per_sample_amortized, n_samples,
                per_target_reference):
    """Total-cost curves for plotting amortized vs reference over M observations."""
    M = np.asarray(M_values, float)
    amort = train_cost + M * n_samples * per_sample_amortized
    ref = M * per_target_reference
    return amort, ref


def source_audit(values, candidates, rtol=0.05):
    """Which candidate method produced `values`? candidates: {name: array}.

    Returns the matching name, or 'unknown'. Use to assert a reported row belongs
    to the method it claims to (item 28).
    """
    v = np.asarray(values, float)
    for name, arr in candidates.items():
        a = np.asarray(arr, float)
        if a.shape == v.shape and np.allclose(v, a, rtol=rtol):
            return name
    return "unknown"


def relative_gap(prism_value, best_value, lower_is_better=True):
    """Signed relative gap of PRISM vs the best competitor (item 29)."""
    if lower_is_better:
        return float((prism_value - best_value) / (abs(best_value) + 1e-12))
    return float((best_value - prism_value) / (abs(best_value) + 1e-12))


# task -> which metric is meaningful as the primary generality score
_MULTIMODAL = {"two_moons"}
_POINT = {"darcy", "burgers", "helmholtz", "gaussian_linear", "oscillator",
          "lotka_volterra", "slcp"}


def primary_metric(task):
    """Point tasks -> inverse R^2; disconnected multimodal -> a calibration metric."""
    if task in _MULTIMODAL:
        return "c2st"        # posterior-appropriate; R^2 is meaningless here
    return "inverse_r2"
