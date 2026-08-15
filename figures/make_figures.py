"""Generate all paper figures (F1-F13) from cached results/ into paper_assets/.

Reads only results/<phase>.json|npz (except the F1 schematic, which is drawn).
Run after the experiment phases:  python figures/make_figures.py
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prism.utils.plotting import set_style, save_fig, PALETTE, color_for  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper_assets")


def _json(phase):
    p = os.path.join(RESULTS, f"{phase}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _npz(phase):
    p = os.path.join(RESULTS, f"{phase}.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


# --------------------------------------------------------------------- F1
def f1_architecture():
    set_style()
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    linewidth=1.2, edgecolor=color, facecolor=color + "22"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2, text="", color="#444444"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.0, color=color))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text, ha="center", fontsize=7, color=color)

    box(0.2, 2.4, 1.6, 1.0, "observation\n$y$", PALETTE[5])
    box(0.2, 0.5, 1.6, 1.0, "parameters\n$x$", PALETTE[0])
    box(2.4, 2.4, 1.8, 1.0, "embed\n$\\mu(y),\\sigma(y)$", PALETTE[5])
    box(2.4, 0.5, 1.8, 1.0, "constraint\nencode", PALETTE[2])
    box(4.8, 1.3, 2.2, 1.3, "liquid-ODE flow\n$z=F(x)$ (exact inv.)", PALETTE[0])
    box(7.6, 2.4, 2.0, 1.0, "base $z\\sim$\n$\\mathcal{N}(\\mu,\\sigma)$", PALETTE[5])
    box(7.6, 0.5, 2.0, 1.0, "project +\nforward head", PALETTE[2])

    arrow(1.8, 2.9, 2.4, 2.9)
    arrow(1.8, 1.0, 2.4, 1.0)
    arrow(4.2, 1.0, 5.0, 1.6, "")
    arrow(4.2, 2.9, 5.4, 2.6, "")
    arrow(7.0, 2.0, 7.6, 2.6, "forward / sample", PALETTE[1])
    arrow(7.0, 1.6, 7.6, 1.0, "inverse $F^{-1}$", PALETTE[0])
    ax.text(5.9, 0.15, "exact reversibility + hard constraints + calibrated posterior",
            ha="center", fontsize=7.5, style="italic", color="#333333")
    return save_fig(fig, "F1_architecture", OUT)


# --------------------------------------------------------------------- F2
def f2_fields():
    d = _npz("phase2")
    if d is None:
        return None
    tasks = ["darcy", "burgers", "helmholtz"]
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for j, t in enumerate(tasks):
        tk, pk = f"F2_{t}_true", f"F2_{t}_pred"
        if tk not in d:
            continue
        true, pred = d[tk], d[pk]
        if true.ndim == 2:
            axes[0, j].imshow(true, cmap="viridis"); axes[1, j].imshow(pred, cmap="viridis")
            axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
            axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
        else:
            xs = np.linspace(0, 1, len(true))
            axes[0, j].plot(xs, true, color=PALETTE[7]); axes[0, j].grid(alpha=.3)
            axes[1, j].plot(xs, pred, color=PALETTE[0]); axes[1, j].grid(alpha=.3)
        axes[0, j].set_title(t.capitalize())
    axes[0, 0].set_ylabel("ground truth")
    axes[1, 0].set_ylabel("PRISM recovered")
    fig.suptitle("Field recovery: ground truth vs PRISM", fontsize=10)
    return save_fig(fig, "F2_fields", OUT)


# --------------------------------------------------------------------- F3
def f3_cycle():
    r = _json("phase2")
    if not r:
        return None
    tasks = list(r["T3"].keys()); methods = ["PRISM", "PRISM-soft", "cINN"]
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    x = np.arange(len(tasks)); w = 0.26
    for k, m in enumerate(methods):
        vals = [r["T3"][t][m] for t in tasks]
        ax.bar(x + (k - 1) * w, vals, w, label=m, color=color_for(m, k))
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("cycle-consistency error"); ax.legend()
    ax.set_title("Exact vs soft reversibility")
    return save_fig(fig, "F3_cycle", OUT)


# --------------------------------------------------------------------- F4
def f4_constraints():
    r = _json("phase2")
    if not r:
        return None
    tasks = list(r["T2"].keys()); methods = ["PRISM", "cINN", "NPE-CNF", "VI"]
    set_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(tasks)); w = 0.2
    for k, m in enumerate(methods):
        vals = [100 * r["T2"][t][m] for t in tasks]
        ax.bar(x + (k - 1.5) * w, vals, w, label=m, color=color_for(m, k))
    ax.set_xticks(x); ax.set_xticklabels([t.replace("_", "\n") for t in tasks])
    ax.set_ylabel("violation rate (\\%)"); ax.legend()
    ax.set_title("Constraint violation (posterior samples)")
    return save_fig(fig, "F4_constraints", OUT)


# --------------------------------------------------------------------- F5
def f5_posteriors():
    d = _npz("phase4")
    if d is None:
        return None
    tasks = ["gaussian_linear", "slcp", "two_moons"]
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for j, t in enumerate(tasks):
        pk = f"F5_{t}_prism"
        if pk not in d:
            axes[j].axis("off"); continue
        post = d[pk]
        axes[j].scatter(post[:, 0], post[:, 1], s=4, alpha=0.3,
                        color=PALETTE[0], label="PRISM")
        rk = f"F5_{t}_ref"
        if rk in d:
            ref = d[rk]
            axes[j].scatter(ref[:, 0], ref[:, 1], s=4, alpha=0.3,
                            color=PALETTE[1], label="reference")
        tr = d.get(f"F5_{t}_true")
        if tr is not None:
            axes[j].scatter(tr[0], tr[1], marker="*", s=120, color="k", zorder=5, label="truth")
        axes[j].set_title(t.replace("_", " ")); axes[j].grid(alpha=.3)
    axes[0].legend(markerscale=2, loc="best")
    fig.suptitle("Posterior samples: PRISM vs reference", fontsize=10)
    return save_fig(fig, "F5_posteriors", OUT)


# --------------------------------------------------------------------- F6
def f6_calibration():
    d = _npz("phase4")
    if d is None:
        return None
    tasks = ["gaussian_linear", "slcp", "two_moons"]
    set_style()
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.4))
    # SBC rank histograms
    for j, t in enumerate(tasks):
        rk = f"F6_{t}_sbc_ranks"
        if rk not in d:
            axes[j].axis("off"); continue
        ranks = d[rk].ravel()
        axes[j].hist(ranks, bins=15, color=PALETTE[0], alpha=0.8,
                     weights=np.ones_like(ranks) / len(ranks) * 15)
        axes[j].axhline(1.0, color="k", ls="--", lw=0.8)
        axes[j].set_title(f"SBC ranks: {t.replace('_',' ')}")
        axes[j].set_xticks([])
    # reliability curve
    ax = axes[3]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for k, t in enumerate(tasks):
        lk, ek = f"F7_{t}_levels", f"F7_{t}_emp"
        if lk in d:
            ax.plot(d[lk], d[ek], "o-", color=color_for(t, k), label=t.replace("_", " "), ms=4)
    ax.set_xlabel("nominal"); ax.set_ylabel("empirical"); ax.set_title("Reliability")
    ax.legend(fontsize=6)
    return save_fig(fig, "F6_calibration", OUT)


# --------------------------------------------------------------------- F7
def f7_coverage():
    d = _npz("phase4")
    if d is None:
        return None
    tasks = ["gaussian_linear", "slcp", "two_moons"]
    set_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    for k, t in enumerate(tasks):
        lk, ek = f"F7_{t}_levels", f"F7_{t}_emp"
        if lk in d:
            ax.plot(d[lk], d[ek], "o-", color=color_for(t, k), label=t.replace("_", " "), ms=4)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title("Coverage vs nominal"); ax.legend()
    return save_fig(fig, "F7_coverage", OUT)


# --------------------------------------------------------------------- F8
def f8_superres():
    d = _npz("phase3")
    if d is None:
        return None
    tasks = ["burgers", "helmholtz", "darcy"]
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for k, t in enumerate(tasks):
        fk, ek = f"F8_{t}_factors", f"F8_{t}_relL2"
        if fk in d:
            ax.plot(d[fk], d[ek], "o-", color=color_for(t, k), label=t, ms=5)
    ax.set_xlabel("resolution factor (x train)"); ax.set_ylabel("forward rel-$L_2$")
    ax.set_title("Zero-shot super-resolution"); ax.legend()
    ax.set_xticks([1, 2, 4])
    return save_fig(fig, "F8_superres", OUT)


# --------------------------------------------------------------------- F9
def f9_robustness():
    d = _npz("phase6")
    if d is None or "F9_sigma" not in d:
        return None
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(d["F9_sigma"], d["F9_inverse_R2"], "o-", color=PALETTE[0], label="inverse $R^2$")
    ax2 = ax.twinx()
    ax2.plot(d["F9_sigma"], d["F9_ece"], "s--", color=PALETTE[1], label="ECE")
    ax2.set_ylabel("expected coverage error", color=PALETTE[1]); ax2.grid(False)
    ax.set_xlabel("observation noise $\\sigma$"); ax.set_ylabel("inverse $R^2$", color=PALETTE[0])
    ax.set_title("Robustness to observation noise")
    return save_fig(fig, "F9_robustness", OUT)


# --------------------------------------------------------------------- F10
def f10_ood():
    r = _json("phase6")
    if not r or "F10_in_dist_R2" not in r:
        return None
    set_style()
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    vals = [r["F10_in_dist_R2"], r["F10_ood_R2"]]
    ax.bar([0, 1], vals, color=[PALETTE[0], PALETTE[1]], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["in-dist", "OOD"])
    ax.set_ylabel("inverse $R^2$"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("OOD generalization")
    return save_fig(fig, "F10_ood", OUT)


# --------------------------------------------------------------------- F11
def f11_pareto():
    d = _npz("phase6")
    if d is None or "F11_names" not in d:
        return None
    set_style()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    names = [str(x) for x in d["F11_names"]]
    lat, acc = d["F11_latency"], d["F11_accuracy"]
    for k, nm in enumerate(names):
        ax.scatter(lat[k], acc[k], s=70, color=color_for(nm, k), zorder=3)
        ax.annotate(nm, (lat[k], acc[k]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("latency (ms/sample, log)")
    ax.set_ylabel("inverse $R^2$"); ax.set_title("Accuracy vs inference cost")
    return save_fig(fig, "F11_pareto", OUT)


# --------------------------------------------------------------------- F12
def f12_ablation():
    d = _npz("phase6")
    if d is None or "F12_labels" not in d:
        return None
    set_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    labels = [str(x) for x in d["F12_labels"]]
    dR2 = d["F12_dR2"]
    colors = [PALETTE[0] if v >= 0 else PALETTE[1] for v in dR2]
    ax.barh(range(len(labels)), dR2, color=colors)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("$\\Delta$ inverse $R^2$ vs full PRISM")
    ax.axvline(0, color="k", lw=0.6); ax.set_title("Ablation contributions")
    return save_fig(fig, "F12_ablation", OUT)


# --------------------------------------------------------------------- F13
def f13_theory():
    d = _npz("phase2"); r = _json("phase2")
    if d is None or "F13_solver_steps" not in d:
        return None
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(d["F13_solver_steps"], d["F13_cycle_err"], "o-", color=PALETTE[0])
    axes[0].set_xlabel("solver steps"); axes[0].set_ylabel("invertibility error")
    axes[0].set_yscale("log"); axes[0].set_title("Invertibility vs solver tol")
    # stability: Lipschitz across eps
    rep = (r or {}).get("F13_stability", {})
    eps = [k for k in rep.keys() if k not in ("bounded",)]
    if eps:
        xs = [float(e) for e in eps]
        ys = [rep[e]["mean_lipschitz"] for e in eps]
        ymax = [rep[e]["max_lipschitz"] for e in eps]
        order = np.argsort(xs); xs = np.array(xs)[order]
        axes[1].plot(xs, np.array(ys)[order], "o-", color=PALETTE[0], label="mean")
        axes[1].plot(xs, np.array(ymax)[order], "s--", color=PALETTE[1], label="max")
        axes[1].set_xscale("log"); axes[1].set_xlabel("perturbation $\\epsilon$")
        axes[1].set_ylabel("empirical Lipschitz"); axes[1].legend()
        axes[1].set_title("Empirical stability bound")
    return save_fig(fig, "F13_theory", OUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    funcs = [f1_architecture, f2_fields, f3_cycle, f4_constraints, f5_posteriors,
             f6_calibration, f7_coverage, f8_superres, f9_robustness, f10_ood,
             f11_pareto, f12_ablation, f13_theory]
    made = []
    for fn in funcs:
        try:
            p = fn()
            if p:
                made.append(os.path.basename(p)); print(f"  wrote {os.path.basename(p)}")
            else:
                print(f"  skipped {fn.__name__} (no cached results)")
        except Exception as e:
            import traceback
            print(f"  ERROR {fn.__name__}: {e}\n{traceback.format_exc().splitlines()[-1]}")
    print(f"\n{len(made)} figures -> {OUT}")
    return made


if __name__ == "__main__":
    main()
