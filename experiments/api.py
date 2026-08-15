"""Integration adapter: one place that bridges the revision scripts to the repo API.

The revision experiment scripts call a small, uniform interface:
    make_method(name, task, cfg, **flags)
    get_dataset(task, seed=...)            -> Dataset with .X_train/.y_train/.X_test/.y_test
    reference_posterior(task, y, n)        -> reference posterior samples
    emcee_reference(task, y, n)            -> alias of reference_posterior
    mcmc_reference(task, y, n)             -> timed MCMC reference samples

Internally these translate to the real repo API:
    experiments._common.make_method(name, cfg, constraint=..., **over)
    benchmarks.registry.get_dataset(task, seed=...)   (fields .X_*/.Y_*)
    Dataset.reference_posterior(y, n, seed)
    baselines.mcmc_reference.MCMCReference(dataset).predict_posterior(y, n)

CONFIRM the two maps below against your code, then everything runs unchanged.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# task -> physics-mandated constraint passed to make_method / PRISM
CONSTRAINT_FOR = {
    "oscillator": "positive", "lotka_volterra": "positive",
    "two_moons": "box", "slcp": "box",
    "darcy": "none", "burgers": "none", "helmholtz": "none",
    "gaussian_linear": "none",
}


def _wrap_dataset(ds):
    # expose lowercase y-aliases the revision scripts expect
    ds.y = getattr(ds, "Y", None)
    for split in ("train", "val", "test"):
        Y = getattr(ds, f"Y_{split}", None)
        if Y is not None:
            setattr(ds, f"y_{split}", Y)
    return ds


def get_dataset(task, seed=0, **kw):
    from benchmarks.registry import get_dataset as _gd
    return _wrap_dataset(_gd(task, seed=seed, **kw))


def make_method(name, task, cfg, **flags):
    """Translate (name, task, cfg, **flags) to make_method(name, cfg, constraint, **over).

    Ablation flags map to the repo's variant names / PRISM kwargs:
      liquid=False        -> name 'PRISM-static'
      cond_base/cond_velocity/use_projection/base/n_mix -> passed through to PRISM
    """
    from experiments._common import make_method as _mm
    constraint = CONSTRAINT_FOR.get(task, "none")
    if name == "PRISM" and flags.pop("liquid", True) is False:
        name = "PRISM-static"
    else:
        flags.pop("liquid", None)
    return _mm(name, cfg, constraint=constraint, **flags)


def reference_posterior(task, y, n=2000, seed=0):
    return get_dataset(task, seed=seed).reference_posterior(y, n=n, seed=seed)


# the revision scripts sometimes call emcee_reference(task, y[, n]); same thing here
def emcee_reference(task, y, n=1000, seed=0):
    try:
        return reference_posterior(task, y, n=n, seed=seed)
    except NotImplementedError:
        return None


def mcmc_reference(task, y, n_samples=1000, seed=0):
    from baselines.mcmc_reference import MCMCReference
    ds = get_dataset(task, seed=seed)
    return MCMCReference(ds).predict_posterior(y, n_samples=n_samples, seed=seed)
