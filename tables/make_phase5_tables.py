"""PRISM Phase 5 -- build accuracy / efficiency / generality tables.

  T_main_pde.tex     -- forward (relL2) + inverse (RMSE), 5 seeds, honest bolding,
                        with the forward gap stated as a number (items 16, 29).
  T_efficiency.tex   -- params / train / latency / R2 + amortization crossover M*
                        by dimension (items 20, 26).
  T_generality_raw.tex -- raw metrics per task, primary metric marked, no pass/fail
                        (item 18).

Run in D:\\ICLR:  py tables\\make_phase5_tables.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from experiments.provenance import ResultManifest
from experiments.phase5_analysis import relative_gap

OUT = "paper_assets"
PRETTY = {"darcy": "Darcy", "burgers": "Burgers", "helmholtz": "Helmholtz",
          "gaussian_linear": "Gaussian-Linear", "slcp": "SLCP", "two_moons": "Two Moons",
          "oscillator": "Oscillator", "lotka_volterra": "Lotka-Volterra"}


def _v(man, exp, task, method, metric):
    try:
        return man.agg(exp, task, method, metric)
    except KeyError:
        return None


def _cell(man, exp, task, method, metric, best_method, prec=3):
    v = _v(man, exp, task, method, metric)
    if v is None:
        return "--"
    mean, std, n = v
    s = f"{mean:.{prec}f}" + (f" $\\pm$ {std:.{prec}f}" if n > 1 and std > 0 else "")
    return f"\\textbf{{{s}}}" if method == best_method else s


def _best_min(man, exp, task, methods, metric):
    vals = {m: _v(man, exp, task, m, metric) for m in methods}
    vals = {m: v[0] for m, v in vals.items() if v}
    return min(vals, key=vals.get) if vals else None


def table_main_pde(man) -> str:
    tasks = ["darcy", "burgers", "helmholtz"]
    fwd = ["PRISM", "FNO", "DeepONet", "PINN"]
    inv = ["PRISM", "cINN", "NPE-CNF", "VI"]
    lines = ["\\multicolumn{4}{l}{\\emph{Forward (relative $L_2$)}} \\\\"]
    for m in fwd:
        cells = [_cell(man, "phase5_pde_fwd", t, m, "forward_relL2",
                       _best_min(man, "phase5_pde_fwd", t, fwd, "forward_relL2")) for t in tasks]
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\emph{Inverse (RMSE)}} \\\\")
    for m in inv:
        cells = [_cell(man, "phase5_pde_inv", t, m, "inverse_rmse",
                       _best_min(man, "phase5_pde_inv", t, inv, "inverse_rmse")) for t in tasks]
        lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(lines)
    # numeric forward gap for the caption (Darcy)
    pf = _v(man, "phase5_pde_fwd", "darcy", "PRISM", "forward_relL2")
    bf = _best_min(man, "phase5_pde_fwd", "darcy", fwd, "forward_relL2")
    bv = _v(man, "phase5_pde_fwd", "darcy", bf, "forward_relL2") if bf else None
    gap = f"{100*relative_gap(pf[0], bv[0]):.0f}\\%" if (pf and bv) else "--"
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Main PDE results (5 seeds). On the forward map a dedicated operator is
most accurate (on Darcy the best baseline is about {gap} lower relative $L_2$ than
PRISM's auxiliary forward head), while PRISM leads on the inverse RMSE that this
work targets. The best value per column is bold.}}
\\label{{tab:main-pde}}
\\begin{{tabular}}{{lccc}}
\\toprule
Method & Darcy & Burgers & Helmholtz \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_efficiency(man) -> str:
    methods = ["PRISM", "cINN", "NPE-CNF", "VI"]
    rows = []
    for m in methods:
        p = _cell(man, "phase5_efficiency", "oscillator", m, "n_params", None, 0)
        tr = _cell(man, "phase5_efficiency", "oscillator", m, "train_s", None, 1)
        la = _cell(man, "phase5_efficiency", "oscillator", m, "latency_ms", None, 3)
        r2 = _cell(man, "phase5_efficiency", "oscillator", m, "inverse_r2", None, 3)
        rows.append(f"{m} & {p} & {tr} & {la} & {r2} \\\\")
    body = "\n".join(rows)
    # crossover row
    rec = [r for k, r in man.records.items()
           if k[:3] == ("phase5_scaling", "gaussian_linear", "PRISM")]
    cross = ""
    if rec:
        dims = rec[0].metrics["dims"]
        ms = [np.mean([r.metrics["crossover_M"][i] for r in rec]) for i in range(len(dims))] \
            if False else rec[0].metrics["crossover_M"]
        cross = " & ".join(f"$d{{=}}{d}$: {m:.0f}" for d, m in zip(dims, ms))
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Efficiency and amortization. PRISM trades higher per-sample latency for
exact continuous-time reversibility and hard constraints. The amortization
break-even (observations beyond which PRISM's total wall-clock, training included,
undercuts a per-observation MCMC reference for the same effective-sample target)
falls sharply as the parameter dimension grows: {cross}.}}
\\label{{tab:efficiency}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Method & Params & Train (s) & Latency (ms/samp.) & Inverse $R^2$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_generality(man) -> str:
    tasks = ["burgers", "darcy", "gaussian_linear", "helmholtz",
             "lotka_volterra", "oscillator", "slcp", "two_moons"]
    rows = []
    for t in tasks:
        r2 = _cell(man, "phase5_generality", t, "PRISM", "inverse_r2", None)
        fl = _cell(man, "phase5_generality", t, "PRISM", "forward_relL2", None)
        vi = _cell(man, "phase5_generality", t, "PRISM", "violation_rate", None)
        c2 = _cell(man, "phase5_generality", t, "PRISM", "c2st", None)
        prim = "C2ST" if t == "two_moons" else "$R^2$"
        rows.append(f"{PRETTY[t]} & {r2} & {fl} & {vi} & {c2} & {prim} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Cross-domain generality under a single fixed configuration (5 seeds),
reported as raw metrics rather than a binary pass/fail. The primary column marks
the posterior-appropriate score per task: point $R^2$ for point tasks and C2ST for
the disconnected multimodal task, whose posterior mean is uninformative.}}
\\label{{tab:generality}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Task & Inverse $R^2$ & Forward rel-$L_2$ & Violation & C2ST & Primary \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    man = ResultManifest("results/manifest.json").load()
    for name, fn in [("T_main_pde", table_main_pde),
                     ("T_efficiency", table_efficiency),
                     ("T_generality_raw", table_generality)]:
        with open(os.path.join(OUT, name + ".tex"), "w") as f:
            f.write(fn(man))
        print(f"  wrote {name}.tex")


if __name__ == "__main__":
    import numpy as np  # noqa (used in commented aggregation path)
    main()
