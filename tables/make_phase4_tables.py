"""PRISM Phase 4 -- build calibration tables from the manifest.

  T_calibration_ext.tex  -- calibration across 5 domains, PRISM/NPE-CNF/VI, with
                            HONEST per-metric bolding (item 24): the true best per
                            column is bolded, even when it is not PRISM.
  T_joint_geometry.tex   -- Two Moons marginal coverage vs joint MMD + mode
                            balance, making the item-19 distinction explicit.

Run in D:\\ICLR:  py tables\\make_phase4_tables.py
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from experiments.provenance import ResultManifest

OUT = "paper_assets"
PRETTY = {
    "gaussian_linear": "Gaussian-Linear", "slcp": "SLCP", "two_moons": "Two Moons",
    "oscillator": "Oscillator", "lotka_volterra": "Lotka-Volterra",
}
# metric -> (target kind).  'p50'=closest to 0.5, 'p90'=closest to 0.9, 'min'=lower better
TARGET = {"c2st": ("p50", 0.5), "cov90": ("p90", 0.9), "ece": ("min", None),
          "nll_phys": ("min", None), "sbc_p": ("max", None)}


def _val(man, exp, task, method, metric):
    try:
        return man.agg(exp, task, method, metric)
    except KeyError:
        return None


def _best_method(man, exp, task, methods, metric):
    kind, tgt = TARGET[metric]
    vals = {m: _val(man, exp, task, m, metric) for m in methods}
    vals = {m: v[0] for m, v in vals.items() if v is not None}
    if not vals:
        return None
    if kind == "min":
        return min(vals, key=lambda m: vals[m])
    if kind == "max":
        return max(vals, key=lambda m: vals[m])
    return min(vals, key=lambda m: abs(vals[m] - tgt))   # p50 / p90


def _cell(man, exp, task, method, metric, best, prec=3):
    v = _val(man, exp, task, method, metric)
    if v is None:
        return "--"
    mean, std, n = v
    s = f"{mean:.{prec}f}" + (f" $\\pm$ {std:.{prec}f}" if n > 1 and std > 0 else "")
    return f"\\textbf{{{s}}}" if method == best else s


def table_calibration(man) -> str:
    exp = "phase4_calibration"
    tasks = ["gaussian_linear", "slcp", "two_moons", "oscillator", "lotka_volterra"]
    methods = ["PRISM", "NPE-CNF", "VI"]
    metrics = [("c2st", "C2ST", 3), ("cov90", "Cov@90", 3),
               ("ece", "ECE", 3), ("nll_phys", "NLL", 2), ("sbc_p", "SBC $p$", 2)]
    lines = []
    for t in tasks:
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{PRETTY[t]}}}}} \\\\")
        best = {mk: _best_method(man, exp, t, methods, mk) for mk, _, _ in metrics}
        for m in methods:
            cells = [_cell(man, exp, t, m, mk, best[mk], pr) for mk, _, pr in metrics]
            lines.append(f"\\quad {m} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    body = "\n".join(lines[:-1])
    header = " & ".join(lbl for _, lbl, _ in metrics)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Calibration across five domains (5 seeds). The best value per task and
metric is shown in bold, including where a baseline wins: PRISM is competitive
with the strongest learned baselines and has strong coverage and C2ST on several
tasks, while variational inference attains the lowest ECE on the Gaussian tasks.
C2ST is best near $0.5$; Cov@90 near $0.9$; ECE and NLL lower-is-better; SBC $p$
higher indicates ranks consistent with uniform.}}
\\label{{tab:calibration-ext}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & {header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def table_joint_geometry(man) -> str:
    exp = "phase4_geometry"
    def c(metric, prec=3):
        v = _val(man, exp, "two_moons", "PRISM", metric)
        if v is None:
            return "--"
        mean, std, n = v
        return f"{mean:.{prec}f}" + (f" $\\pm$ {std:.{prec}f}" if n > 1 and std > 0 else "")
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Two Moons: marginal calibration vs joint geometry. Marginal coverage is
acceptable (low marginal ECE), yet the joint MMD is far larger than the marginal
MMD and the mode balance is low: a single connected base blurs the two crescents.
We therefore do not present marginal calibration as evidence of a correct joint
posterior.}}
\\label{{tab:joint-geometry}}
\\begin{{tabular}}{{lcccc}}
\\toprule
 & Marginal ECE & Marginal MMD & Joint MMD & Mode balance \\\\
\\midrule
PRISM (Gaussian base) & {c('marginal_ece')} & {c('marginal_mmd',4)} & {c('joint_mmd',4)} & {c('mode_balance',2)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    man = ResultManifest("results/manifest.json").load()
    for name, fn in [("T_calibration_ext", table_calibration),
                     ("T_joint_geometry", table_joint_geometry)]:
        with open(os.path.join(OUT, name + ".tex"), "w") as f:
            f.write(fn(man))
        print(f"  wrote {name}.tex")


if __name__ == "__main__":
    main()
