"""PRISM Phase 2 -- build tables from the manifest (single source of truth).

Emits three LaTeX tables to paper_assets/:
  T_cycle_phase2.tex        -- latent (guaranteed) vs physical (empirical) cycle
  T_constraints_fair.tex    -- fair constraint comparison (item 10)
  T_stability.tex           -- Lipschitz factors, bound, empirical sensitivity (9,31)

All values are read from results/manifest.json via ResultManifest, so a table
cell can never disagree with a figure drawn from the same record.

Run in D:\\ICLR:  py tables\\make_phase2_tables.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from experiments.provenance import ResultManifest

OUT = "paper_assets"
PRETTY = {
    "oscillator": "Oscillator", "lotka_volterra": "Lotka-Volterra",
    "two_moons": "Two Moons", "slcp": "SLCP", "darcy": "Darcy",
    "burgers": "Burgers", "helmholtz": "Helmholtz",
}
EPS_HI = "0.1"


def _cell(man, exp, task, method, metric, prec=3, sci=False):
    try:
        mean, std, n = man.agg(exp, task, method, metric)
    except KeyError:
        return "--"
    if sci:
        return f"{mean:.1e}" + (f" $\\pm$ {std:.0e}" if n > 1 and std > 0 else "")
    return f"{mean:.{prec}f}" + (f" $\\pm$ {std:.{prec}f}" if n > 1 and std > 0 else "")


def table_cycle(man) -> str:
    tasks = ["oscillator", "darcy", "burgers"]
    rows = []
    for t in tasks:
        lat = _cell(man, "phase2_reversibility", t, "PRISM", "latent_floor", sci=True)
        order = _cell(man, "phase2_reversibility", t, "PRISM", "latent_order", prec=2)
        phys = _cell(man, "phase2_reversibility", t, "PRISM", "physical_cycle_mean", sci=True)
        rows.append(f"{PRETTY[t]} & {lat} & {order} & {phys} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Reversibility. The \\emph{{latent}} cycle error is limited by the
integration floor and the fitted convergence order matches the integrator
(Proposition~\\ref{{prop:num}}); the \\emph{{physical}} cycle
$\\lVert h(g_\\psi(x))-x\\rVert$ is reported separately as an empirical quantity,
not a guarantee.}}
\\label{{tab:cycle}}
\\begin{{tabular}}{{lccc}}
\\toprule
Task & Latent floor & Fitted order & Physical cycle (mean) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_constraints_fair(man) -> str:
    tasks = ["oscillator", "lotka_volterra", "two_moons", "slcp"]
    methods = ["PRISM", "cINN", "NPE-CNF", "VI"]
    lines = []
    for t in tasks:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{PRETTY[t]}}}}} \\\\")
        for m in methods:
            viol = _cell(man, "phase2_constraints", t, m, "violation_rate", prec=3)
            rmse = _cell(man, "phase2_constraints", t, m, "inverse_rmse", prec=3)
            cov = _cell(man, "phase2_constraints", t, m, "cov90", prec=3)
            nll = _cell(man, "phase2_constraints", t, m, "nll_phys", prec=2)
            lat = _cell(man, "phase2_constraints", t, m, "latency_ms", prec=2)
            lines.append(f"\\quad {m} & {viol} & {rmse} & {cov} & {nll} & {lat} \\\\")
        lines.append("\\midrule")
    body = "\n".join(lines[:-1])
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Fair constraint comparison. Every method receives the \\emph{{same}}
support transform, so the violation rate is $\\approx 0$ for all; the comparison
therefore turns on posterior quality (inverse RMSE, $90\\%$ coverage,
physical-space NLL) and latency rather than feasibility.}}
\\label{{tab:constraints-fair}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & Viol. & Inv. RMSE & Cov@90 & NLL$_{{\\text{{phys}}}}$ & Lat. (ms) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_stability(man) -> str:
    tasks = ["oscillator", "slcp"]
    rows = []
    for t in tasks:
        lc = _cell(man, "phase2_stability", t, "PRISM", "lip_c", prec=2)
        lg = _cell(man, "phase2_stability", t, "PRISM", "L_g", prec=2)
        lm = _cell(man, "phase2_stability", t, "PRISM", "lip_mu", prec=2)
        bd = _cell(man, "phase2_stability", t, "PRISM", "bound", sci=True)
        sm = _cell(man, "phase2_stability", t, "PRISM", f"sens_mean_{EPS_HI}", prec=3)
        sx = _cell(man, "phase2_stability", t, "PRISM", f"sens_max_{EPS_HI}", prec=3)
        rows.append(f"{PRETTY[t]} & {lc} & {lg} & {lm} & {bd} & {sm} & {sx} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Stability. The global Gr\\\"onwall bound
$\\mathrm{{Lip}}(h)\\le\\mathrm{{Lip}}(c)\\,e^{{L_g}}\\,\\mathrm{{Lip}}(\\mu_\\phi)$
is loose; the empirical local sensitivity (mean and max over held-out
observations at $\\epsilon={EPS_HI}$) is far smaller and is what governs practical
robustness. We do not equate the empirical max with the true global constant.}}
\\label{{tab:stability}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
Task & $\\mathrm{{Lip}}(c)$ & $L_g$ & $\\mathrm{{Lip}}(\\mu_\\phi)$ & Bound & Sens. mean & Sens. max \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    man = ResultManifest("results/manifest.json").load()
    for name, fn in [("T_cycle_phase2", table_cycle),
                     ("T_constraints_fair", table_constraints_fair),
                     ("T_stability", table_stability)]:
        path = os.path.join(OUT, name + ".tex")
        with open(path, "w") as f:
            f.write(fn(man))
        print(f"  wrote {name}.tex")


if __name__ == "__main__":
    main()
