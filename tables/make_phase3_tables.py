"""PRISM Phase 3 -- build the four ablation tables from the manifest.

  T_liquid_vs_static.tex  -- per-task liquid vs static (+ paired verdict, params)
  T_conditioning.tex      -- base / dynamics / both (+ collapse diagnostics)
  T_hybrid_velocity.tex   -- shared vs hybrid (+ verdict, params)
  T_projection.tex        -- proj vs no-proj on box tasks (violations jump)

Verdicts come from paired, sign-consistent, effect-size tests over seeds
(experiments/phase3_analysis.py), read from the single manifest.

Run in D:\\ICLR:  py tables\\make_phase3_tables.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from experiments.provenance import ResultManifest
from experiments.phase3_analysis import paired_delta, classify_regime

OUT = "paper_assets"
PRETTY = {
    "darcy": "Darcy", "slcp": "SLCP", "two_moons": "Two Moons",
    "oscillator": "Oscillator", "lotka_volterra": "Lotka-Volterra",
    "gaussian_linear": "Gaussian-Linear",
}


def _series(man, exp, task, method, metric):
    recs = sorted((r for k, r in man.records.items() if k[:3] == (exp, task, method)),
                  key=lambda r: r.seed)
    return [r.metrics[metric] for r in recs if metric in r.metrics]


def _cell(man, exp, task, method, metric, prec=3):
    try:
        mean, std, n = man.agg(exp, task, method, metric)
    except KeyError:
        return "--"
    return f"{mean:.{prec}f}" + (f" $\\pm$ {std:.{prec}f}" if n > 1 and std > 0 else "")


def table_liquid(man) -> str:
    tasks = ["darcy", "slcp", "two_moons", "oscillator", "lotka_volterra"]
    rows = []
    for t in tasks:
        rl = _cell(man, "phase3_liquid", t, "liquid", "inverse_r2")
        rs = _cell(man, "phase3_liquid", t, "static", "inverse_r2")
        el = _cell(man, "phase3_liquid", t, "liquid", "nll_phys", 2)
        es = _cell(man, "phase3_liquid", t, "static", "nll_phys", 2)
        a = _series(man, "phase3_liquid", t, "liquid", "inverse_r2")
        b = _series(man, "phase3_liquid", t, "static", "inverse_r2")
        verdict = "--"
        if len(a) == len(b) and len(a) > 1:
            verdict = classify_regime(paired_delta(a, b), higher_is_better=True)
        rows.append(f"{PRETTY[t]} & {rl} & {rs} & {el} & {es} & {verdict} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Liquid vs static transport across tasks (matched parameter counts,
5 seeds). The verdict is a paired, sign-consistent, effect-size test over seeds;
liquid dynamics help most on the harder posterior geometries rather than
uniformly, which we state plainly.}}
\\label{{tab:liquid-static}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Task & Liquid $R^2$ & Static $R^2$ & Liquid NLL & Static NLL & Verdict \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_conditioning(man) -> str:
    tasks = ["slcp", "two_moons", "gaussian_linear"]
    variants = ["base", "dynamics", "both"]
    lines = []
    for t in tasks:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{PRETTY[t]}}}}} \\\\")
        for v in variants:
            c2 = _cell(man, "phase3_conditioning", t, v, "c2st")
            cov = _cell(man, "phase3_conditioning", t, v, "cov90")
            nll = _cell(man, "phase3_conditioning", t, v, "nll_phys", 2)
            sr = _cell(man, "phase3_conditioning", t, v, "spread_ratio", 2)
            yd = _cell(man, "phase3_conditioning", t, v, "y_dependence", 3)
            lines.append(f"\\quad {v} & {c2} & {cov} & {nll} & {sr} & {yd} \\\\")
        lines.append("\\midrule")
    body = "\n".join(lines[:-1])
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Conditioning ablation. A variant that ignores the observation shows a
spread ratio near or above $1$ and a y-dependence near $0$; conditioning the base
(our default) yields informative, y-dependent posteriors. We report this as a
measured property of our architecture, not a generic optimization law.}}
\\label{{tab:conditioning}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Variant & C2ST & Cov@90 & NLL$_{{\\text{{phys}}}}$ & Spread ratio & y-dependence \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_hybrid(man) -> str:
    tasks = ["gaussian_linear", "slcp"]
    rows = []
    for t in tasks:
        cs = _cell(man, "phase3_hybrid", t, "shared", "c2st")
        ch = _cell(man, "phase3_hybrid", t, "hybrid", "c2st")
        ns = _cell(man, "phase3_hybrid", t, "shared", "n_params", 0)
        nh = _cell(man, "phase3_hybrid", t, "hybrid", "n_params", 0)
        a = _series(man, "phase3_hybrid", t, "hybrid", "c2st")
        b = _series(man, "phase3_hybrid", t, "shared", "c2st")
        verdict = "--"
        if len(a) == len(b) and len(a) > 1:
            # C2ST is lower-is-better (closer to 0.5); use nll instead for direction-safety
            verdict = classify_regime(paired_delta(a, b), higher_is_better=False)
        rows.append(f"{PRETTY[t]} & {cs} & {ch} & {ns} & {nh} & {verdict} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Hybrid conditional velocity vs shared transport. Conditioning the
velocity field adds parameters; the verdict column reports whether it improves
calibration over the shared-transport default. Where it does not, the simpler
design is retained.}}
\\label{{tab:hybrid}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Task & Shared C2ST & Hybrid C2ST & Shared params & Hybrid params & Verdict \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_projection(man) -> str:
    tasks = ["slcp", "two_moons"]
    rows = []
    for t in tasks:
        vp = _cell(man, "phase3_projection", t, "proj", "violation_rate")
        vn = _cell(man, "phase3_projection", t, "noproj", "violation_rate")
        rp = _cell(man, "phase3_projection", t, "proj", "inverse_rmse")
        rn = _cell(man, "phase3_projection", t, "noproj", "inverse_rmse")
        cp = _cell(man, "phase3_projection", t, "proj", "cov90")
        cn = _cell(man, "phase3_projection", t, "noproj", "cov90")
        rows.append(f"{PRETTY[t]} & {vp} & {vn} & {rp} & {rn} & {cp} & {cn} \\\\")
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Projection ablation on box-constrained tasks. Without the projection,
posterior tail samples leave the feasible set (violation rate jumps), while
inverse accuracy and coverage are largely unchanged -- isolating the projection's
role as support satisfaction.}}
\\label{{tab:projection}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{Violation}} & \\multicolumn{{2}}{{c}}{{Inv. RMSE}} & \\multicolumn{{2}}{{c}}{{Cov@90}} \\\\
Task & proj & no-proj & proj & no-proj & proj & no-proj \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    man = ResultManifest("results/manifest.json").load()
    for name, fn in [("T_liquid_vs_static", table_liquid),
                     ("T_conditioning", table_conditioning),
                     ("T_hybrid_velocity", table_hybrid),
                     ("T_projection", table_projection)]:
        with open(os.path.join(OUT, name + ".tex"), "w") as f:
            f.write(fn(man))
        print(f"  wrote {name}.tex")


if __name__ == "__main__":
    main()
