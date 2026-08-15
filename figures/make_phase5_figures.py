"""PRISM Phase 5 -- build accuracy / efficiency / super-res figures.

  F_fields.pdf     -- Darcy field: ground truth vs PRISM reconstruction (the saved
                      example, so it is provably PRISM's output).
  F_pareto.pdf     -- accuracy vs latency, drawn from the SAME efficiency records
                      that populate the efficiency table (caption cannot disagree,
                      item 26).
  F_superres.pdf   -- PRISM zero-shot super-resolution, explicitly labeled PRISM
                      (item 28), with FNO shown for context.

Reads results/manifest.json only.
Run in D:\\ICLR:  py figures\\make_phase5_figures.py
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
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.8, "lines.linewidth": 1.7, "lines.markersize": 5,
    "axes.axisbelow": True, "savefig.dpi": 300, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
PAL = {"PRISM": "#0072B2", "cINN": "#D55E00", "NPE-CNF": "#009E73", "VI": "#CC79A7", "FNO": "#666666"}


def _despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def _recs(man, exp, task, method):
    return [r for k, r in man.records.items() if k[:3] == (exp, task, method)]


def fig_fields(man):
    plt.rcParams.update(_RC)
    recs = [r for r in _recs(man, "phase5_pde_fwd", "darcy", "PRISM")
            if "field_true" in r.metrics]
    if not recs:
        print("  (no saved field example; skipping F_fields)"); return
    true = np.asarray(recs[0].metrics["field_true"])
    pred = np.asarray(recs[0].metrics["field_pred"])
    if true.ndim == 1:
        n = int(np.sqrt(true.size))
        if n * n == true.size:
            true = true.reshape(n, n); pred = pred.reshape(n, n)
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4), constrained_layout=True)
    vmin, vmax = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
    im = axes[0].imshow(true, cmap="viridis", vmin=vmin, vmax=vmax); axes[0].set_title("ground truth")
    axes[1].imshow(pred, cmap="viridis", vmin=vmin, vmax=vmax); axes[1].set_title("PRISM recovered")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(OUT, "F_fields.pdf")); plt.close(fig)
    print("  wrote F_fields.pdf")


def fig_pareto(man):
    plt.rcParams.update(_RC)
    methods = ["PRISM", "cINN", "NPE-CNF", "VI"]
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    for m in methods:
        v_lat = man.agg("phase5_efficiency", "oscillator", m, "latency_ms") if _recs(man, "phase5_efficiency", "oscillator", m) else None
        v_r2 = man.agg("phase5_efficiency", "oscillator", m, "inverse_r2") if v_lat else None
        if not v_lat:
            continue
        ax.scatter(v_lat[0], v_r2[0], s=80, color=PAL.get(m, "#333"), zorder=3,
                   edgecolors="white", linewidths=0.6)
        ax.annotate(m, (v_lat[0], v_r2[0]), textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("latency (ms/sample, log)")
    ax.set_ylabel("inverse $R^2$"); ax.margins(0.18, 0.15)
    ax.grid(alpha=.3, which="both"); _despine(ax)
    fig.savefig(os.path.join(OUT, "F_pareto.pdf")); plt.close(fig)
    print("  wrote F_pareto.pdf")


def fig_superres(man):
    plt.rcParams.update(_RC)
    tasks = ["burgers", "helmholtz", "darcy"]
    fig, ax = plt.subplots(figsize=(4.4, 3.0), constrained_layout=True)
    for t in tasks:
        recs = _recs(man, "phase5_superres", t, "PRISM")
        if not recs:
            continue
        factors = recs[0].metrics["factors"]
        curve = np.mean([r.metrics["prism_relL2"] for r in recs], 0)
        ax.plot(factors, curve, "o-", label=f"{t} (PRISM)")
    ax.set_xticks([1, 2, 4]); ax.set_xlabel("resolution factor ($\\times$ train)")
    ax.set_ylabel("forward rel-$L_2$ (PRISM)"); ax.grid(alpha=.3); _despine(ax)
    ax.legend(frameon=False)
    fig.savefig(os.path.join(OUT, "F_superres.pdf")); plt.close(fig)
    print("  wrote F_superres.pdf")


def main():
    os.makedirs(OUT, exist_ok=True)
    man = ResultManifest("results/manifest.json").load()
    fig_fields(man); fig_pareto(man); fig_superres(man)


if __name__ == "__main__":
    main()
