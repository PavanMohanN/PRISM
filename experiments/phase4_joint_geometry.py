"""PRISM Phase 4 -- E9: Two Moons joint geometry (guide item 19).

Marginal coverage can look fine while the JOINT posterior geometry is wrong. We
make the distinction explicit on Two Moons: report (a) marginal central-interval
coverage (expected to look acceptable), (b) the marginal-vs-joint MMD gap, and
(c) mode coverage against the two reference crescents. We save PRISM and reference
samples so the figure can show the mode-blurring directly. The paper then states
plainly that a single connected base does not recover the disconnected geometry.

Run in D:\\ICLR:  py experiments\\phase4_joint_geometry.py --mode smoke
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
from experiments.phase4_analysis import (
    coverage_curve, ece_from_curve, marginal_joint_gap, mode_coverage,
)

SEEDS = [0, 1, 2, 3, 4]
TASK = "two_moons"


def _reference_posterior(task, y0, n):
    """Reference samples for the task's posterior. Hook to your registry."""
    from experiments.api import reference_posterior  # add if missing
    return np.asarray(reference_posterior(task, y0, n))


def _mode_centers(ref_samples, k=2):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(ref_samples)
    return km.cluster_centers_


def run(seed, cfg) -> dict:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = get_dataset(TASK, seed=seed)
    dim = ds.X_train.shape[-1]
    est = make_method("PRISM", TASK, cfg)
    est.fit(ds.X_train, ds.y_train)

    y0 = ds.y_test[0]
    post = np.asarray(est.predict_posterior(y0, 2000)).reshape(-1, dim)
    ref = _reference_posterior(TASK, y0, 2000)

    # marginal coverage (looks acceptable)
    n = min(64, len(ds.y_test))
    truths = np.asarray(ds.X_test[:n])
    postN = np.stack([np.asarray(est.predict_posterior(ds.y_test[i], 200)).reshape(-1, dim)[:200]
                      for i in range(n)], 0)
    levels, emp = coverage_curve(truths, postN, (0.5, 0.8, 0.9))
    marg_ece = ece_from_curve(levels, emp)

    # joint geometry (wrong): marginal vs joint gap + mode coverage
    marg_mmd, joint_mmd = marginal_joint_gap(post, ref)
    centers = _mode_centers(ref, 2)
    _, balance = mode_coverage(post, centers)

    return {"marginal_ece": float(marg_ece),
            "marginal_mmd": marg_mmd, "joint_mmd": joint_mmd,
            "mode_balance": float(balance),
            "post_xy": post[:800].tolist(), "ref_xy": ref[:800].tolist()}


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
    man.run_seeds("phase4_geometry", TASK, "PRISM",
                  fn=lambda s: run(s, cfg), seeds=SEEDS, config={"mode": args.mode})
    me = man.agg("phase4_geometry", TASK, "PRISM", "marginal_ece")[0]
    jm = man.agg("phase4_geometry", TASK, "PRISM", "joint_mmd")[0]
    mb = man.agg("phase4_geometry", TASK, "PRISM", "mode_balance")[0]
    print(f"  [two_moons] marginal ECE={me:.3f} (looks fine)  joint MMD={jm:.3f}  "
          f"mode balance={mb:.2f} (low => blurred)")
    man.save()
    print(f"saved -> {args.manifest}")


if __name__ == "__main__":
    main()
