"""PRISM Phase 2 -- build figures from the manifest.

  F_reversibility.pdf -- log-log latent & physical cycle vs solver steps, with the
                         fitted convergence order annotated and the floor marked
                         (fixes the flat-curve objection, item 21).
  F_stability.pdf     -- empirical local sensitivity (mean & max) vs perturbation
                         eps, with the loose global bound shown for context (31).

Reads results/manifest.json only; nothing is recomputed here.
Run in D:\\ICLR:  py figures\\make_phase2_figures.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import numpy as np
import matplotlib.pyplot as plt

from experiments.provenance import ResultManifest
from experiments.phase2_analysis import loglog_order_and_floor

OUT = "paper_assets"
_RC = {
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.7, "lines.markersize": 4.5,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def _despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _records(man, exp, task, method):
    return [r for k, r in man.records.items() if k[:3] == (exp, task, method)]


def _mean_curve(man, exp, task, key):
    recs = _records(man, exp, task, "PRISM")
    arrs = np.array([r.metrics[key] for r in recs if key in r.metrics])
    return arrs.mean(0) if len(arrs) else None


def fig_reversibility(man):
    plt.rcParams.update(_RC)
    tasks = ["oscillator", "darcy", "burgers"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(7.4, 2.7), constrained_layout=True)
    for j, t in enumerate(tasks):
        recs = _records(man, "phase2_reversibility", t, "PRISM")
        if not recs:
            axes[j].axis("off"); continue
        steps = np.asarray(recs[0].metrics["steps"], float)
        lat = _mean_curve(man, "phase2_reversibility", t, "latent_cycle_curve")
        phys = _mean_curve(man, "phase2_reversibility", t, "physical_cycle_curve")
        h = 1.0 / steps
        axes[j].loglog(h, lat, "o-", color=BLUE, label="latent")
        axes[j].loglog(h, phys, "s--", color=ORANGE, label="physical")
        fit = loglog_order_and_floor(steps, lat)
        if np.isfinite(fit["order"]) and fit["n_fit"] >= 2:
            axes[j].annotate(f"order $\\approx$ {fit['order']:.1f}",
                             xy=(0.5, 0.08), xycoords="axes fraction", fontsize=7.5)
            axes[j].axhline(fit["floor"], color="0.6", lw=0.8, ls=":")
        axes[j].set_title(t.capitalize())
        axes[j].set_xlabel("step size $h$")
        _despine(axes[j])
    axes[0].set_ylabel("cycle error")
    axes[0].legend(frameon=False, loc="upper left")
    path = os.path.join(OUT, "F_reversibility.pdf")
    fig.savefig(path); plt.close(fig); print("  wrote F_reversibility.pdf")


def fig_stability(man):
    plt.rcParams.update(_RC)
    tasks = ["oscillator", "slcp"]
    eps = [1e-3, 1e-2, 1e-1]
    fig, axes = plt.subplots(1, len(tasks), figsize=(6.6, 2.7), constrained_layout=True)
    for j, t in enumerate(tasks):
        recs = _records(man, "phase2_stability", t, "PRISM")
        if not recs:
            axes[j].axis("off"); continue
        mean = [np.mean([r.metrics[f"sens_mean_{e:g}"] for r in recs]) for e in eps]
        mx = [np.mean([r.metrics[f"sens_max_{e:g}"] for r in recs]) for e in eps]
        bound = np.mean([r.metrics["bound"] for r in recs])
        axes[j].plot(eps, mean, "o-", color=BLUE, label="empirical mean")
        axes[j].plot(eps, mx, "s--", color=ORANGE, label="empirical max")
        axes[j].axhline(bound, color=GREEN, lw=1.0, ls=":", label="global bound")
        axes[j].set_xscale("log"); axes[j].set_yscale("log")
        axes[j].set_title(t.replace("_", " ").upper() if t == "slcp" else t.capitalize())
        axes[j].set_xlabel("perturbation $\\epsilon$")
        _despine(axes[j])
    axes[0].set_ylabel("local sensitivity ratio")
    axes[0].legend(frameon=False, fontsize=6.5, loc="best")
    path = os.path.join(OUT, "F_stability.pdf")
    fig.savefig(path); plt.close(fig); print("  wrote F_stability.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    man = ResultManifest("results/manifest.json").load()
    fig_reversibility(man)
    fig_stability(man)


if __name__ == "__main__":
    main()
