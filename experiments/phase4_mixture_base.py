"""PRISM Phase 4 -- E10 (OPTIONAL, D6): mixture base for multimodality (item 19).

Optional upside experiment. A unimodal base cannot split disconnected modes; a
Gaussian-mixture base can. We compare the default single-Gaussian base against a
K-component mixture base on Two Moons (and optionally SLCP), measuring mode
coverage against the reference crescents. If the mixture recovers both modes it
earns a multimodal clause back in the abstract; if only partially, we report it
honestly. Skip this file if you did not opt in at the D6 gate.

Hook: the mixture base is selected via a constructor flag, e.g.
make_method("PRISM", task, cfg, base="mixture", n_mix=2). Adjust below to your API.

Run in D:\\ICLR:  py experiments\\phase4_mixture_base.py --mode smoke
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch

from experiments.provenance import ResultManifest
from experiments._common import load_cfg
from experiments.api import make_method
from experiments.api import get_dataset
from experiments.phase4_analysis import mode_coverage, marginal_joint_gap

TASKS = ["two_moons"]
VARIANTS = {"gaussian": dict(base="gaussian"), "mixture": dict(base="mixture", n_mix=2)}
SEEDS = [0, 1, 2, 3, 4]


def _reference_posterior(task, y0, n):
    from experiments.api import reference_posterior
    return np.asarray(reference_posterior(task, y0, n))


def _mode_centers(ref, k=2):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=5, random_state=0).fit(ref).cluster_centers_


def _make(task, cfg, **flags):
    try:
        return make_method("PRISM", task, cfg, **flags)
    except TypeError:
        from prism.models.prism import PRISM
        base = dict(cfg.get("prism", {})) if hasattr(cfg, "get") else {}
        base.update(flags)
        return PRISM(**base)


def run(task, variant, seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(task, seed=seed)
    dim = ds.X_train.shape[-1]
    est = _make(task, cfg, **VARIANTS[variant])
    est.fit(ds.X_train, ds.y_train)

    y0 = ds.y_test[0]
    post = np.asarray(est.predict_posterior(y0, 2000)).reshape(-1, dim)
    ref = _reference_posterior(task, y0, 2000)
    centers = _mode_centers(ref, 2)
    _, balance = mode_coverage(post, centers)
    marg_mmd, joint_mmd = marginal_joint_gap(post, ref)
    return {"mode_balance": float(balance), "joint_mmd": joint_mmd,
            "post_xy": post[:800].tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke")
    ap.add_argument("--manifest", default="results/manifest.json")
    args = ap.parse_args()
    cfg = load_cfg("phase4", args.mode)

    import os
    man = ResultManifest(args.manifest)
    if os.path.exists(args.manifest):
        man.load()
    for task in TASKS:
        for variant in VARIANTS:
            man.run_seeds("phase4_mixture", task, variant,
                          fn=lambda s, v=variant, t=task: run(t, v, s, cfg),
                          seeds=SEEDS, config={"mode": args.mode})
        bg = man.agg("phase4_mixture", task, "gaussian", "mode_balance")[0]
        bm = man.agg("phase4_mixture", task, "mixture", "mode_balance")[0]
        print(f"  [{task}] mode balance  gaussian={bg:.2f}  mixture={bm:.2f}")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
