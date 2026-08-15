"""PRISM Phase 4 -- build calibration and posterior-geometry figures.

  F_calibration_ext.pdf -- SBC rank histograms + reliability curves across all
                           five domains (item 17).
  F_posteriors_tm.pdf   -- Two Moons: reference vs PRISM (Gaussian base) and, if
                           the optional mixture experiment was run, the mixture
                           base, with the mode-balance annotation (item 19).

Reads results/manifest.json only.
Run in D:\\ICLR:  py figures\\make_phase4_figures.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt

from experiments.provenance import ResultManifest

OUT = "paper_assets"
_RC = {
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6, "lines.markersize": 4,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
PRETTY = {"gaussian_linear": "Gaussian-Linear", "slcp": "SLCP", "two_moons": "Two Moons",
          "oscillator": "Oscillator", "lotka_volterra": "Lotka-Volterra"}


def _despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _recs(man, exp, task, method):
    return [r for k, r in man.records.items() if k[:3] == (exp, task, method)]


def fig_calibration(man):
    plt.rcParams.update(_RC)
    tasks = ["gaussian_linear", "slcp", "two_moons", "oscillator", "lotka_volterra"]
    fig, axes = plt.subplots(2, len(tasks), figsize=(2.0 * len(tasks), 4.0),
                             constrained_layout=True)
    for j, t in enumerate(tasks):
        recs = _recs(man, "phase4_calibration", t, "PRISM")
        if not recs:
            axes[0, j].axis("off"); axes[1, j].axis("off"); continue
        ranks = np.concatenate([np.asarray(r.metrics["sbc_ranks"]) for r in recs])
        axes[0, j].hist(ranks / (ranks.max() + 1e-9), bins=12, color=BLUE, alpha=0.85,
                        weights=np.ones_like(ranks) / len(ranks) * 12)
        axes[0, j].axhline(1.0, color="k", ls="--", lw=0.8)
        axes[0, j].set_title(PRETTY[t]); axes[0, j].set_xticks([])
        _despine(axes[0, j])
        lv = np.asarray(recs[0].metrics["rel_levels"])
        emp = np.mean([r.metrics["rel_emp"] for r in recs], 0)
        axes[1, j].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[1, j].plot(lv, emp, "o-", color=ORANGE)
        axes[1, j].set_xlim(0, 1); axes[1, j].set_ylim(0, 1)
        axes[1, j].set_xlabel("nominal"); _despine(axes[1, j])
    axes[0, 0].set_ylabel("SBC density"); axes[1, 0].set_ylabel("empirical")
    fig.savefig(os.path.join(OUT, "F_calibration_ext.pdf")); plt.close(fig)
    print("  wrote F_calibration_ext.pdf")


def fig_posteriors_tm(man):
    plt.rcParams.update(_RC)
    grecs = _recs(man, "phase4_geometry", "two_moons", "PRISM")
    if not grecs:
        print("  (no two_moons geometry records; skipping F_posteriors_tm)"); return
    ref = np.asarray(grecs[0].metrics["ref_xy"])
    post = np.asarray(grecs[0].metrics["post_xy"])
    bal = np.mean([r.metrics["mode_balance"] for r in grecs])

    mix = _recs(man, "phase4_mixture", "two_moons", "mixture")
    npan = 3 if mix else 2
    fig, axes = plt.subplots(1, npan, figsize=(2.4 * npan, 2.5), constrained_layout=True)
    axes[0].scatter(ref[:, 0], ref[:, 1], s=4, alpha=0.35, color="0.5")
    axes[0].set_title("reference")
    axes[1].scatter(post[:, 0], post[:, 1], s=4, alpha=0.35, color=BLUE)
    axes[1].set_title(f"PRISM (Gaussian base)\nmode balance {bal:.2f}")
    if mix:
        pm = np.asarray(mix[0].metrics["post_xy"])
        bm = np.mean([r.metrics["mode_balance"] for r in mix])
        axes[2].scatter(pm[:, 0], pm[:, 1], s=4, alpha=0.35, color=GREEN)
        axes[2].set_title(f"PRISM (mixture base)\nmode balance {bm:.2f}")
    for ax in axes:
        ax.set_xlabel("$x_1$"); _despine(ax)
    axes[0].set_ylabel("$x_2$")
    fig.savefig(os.path.join(OUT, "F_posteriors_tm.pdf")); plt.close(fig)
    print("  wrote F_posteriors_tm.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    man = ResultManifest("results/manifest.json").load()
    fig_calibration(man)
    fig_posteriors_tm(man)


if __name__ == "__main__":
    main()
