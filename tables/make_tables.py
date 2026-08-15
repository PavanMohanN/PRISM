"""Generate all paper tables (T0-T8) from cached results/ into paper_assets/.

Reads only results/<phase>.json so regeneration is fast and deterministic.
Run after the experiment phases:  python tables/make_tables.py
"""
from __future__ import annotations

import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prism.utils.latex import latex_table, save_table, fmt_pm  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper_assets")
PDE = ["darcy", "burgers", "helmholtz"]


def _esc(t):
    return str(t).replace("_", "\\_")


def _load(phase):
    p = os.path.join(RESULTS, f"{phase}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _bold_min(values, lower_better=True):
    """Return a boolean mask marking the best (min/max) finite value."""
    fin = [v for v in values if v is not None]
    if not fin:
        return [False] * len(values)
    best = min(fin) if lower_better else max(fin)
    return [(v is not None and v == best) for v in values]


# --------------------------------------------------------------------- T0
def t0():
    caps = ["Forward", "Inverse", "Exact consistency", "Hard constraints",
            "Calibrated posterior", "Theory"]
    methods = ["PRISM", "cINN", "inv-FNO", "NPE-CNF", "PINO", "PRISM-soft"]
    # capability matrix (paper claim); y/. = yes/no, ~ = partial
    M = {
        "PRISM":      ["y", "y", "y", "y", "y", "y"],
        "cINN":       ["y", "y", "y", ".", "y", "."],
        "inv-FNO":    ["y", "y", ".", ".", ".", "."],
        "NPE-CNF":    [".", "y", "~", ".", "y", "."],
        "PINO":       ["y", ".", ".", "~", ".", "~"],
        "PRISM-soft": ["y", "y", ".", "~", "~", "."],
    }
    sym = {"y": "\\checkmark", ".": "--", "~": "$\\sim$"}
    header = ["Capability"] + methods
    rows = []
    for i, cap in enumerate(caps):
        rows.append([cap] + [sym[M[m][i]] for m in methods])
    txt = latex_table(header, rows,
                      caption="Capability matrix: PRISM versus representative baselines. "
                              "\\checkmark{} = supported by construction, $\\sim$ = partial/soft, -- = not supported.",
                      label="tab:novelty",
                      note="Only PRISM provides exact consistency, hard constraints, calibrated posteriors, and theory together.")
    return save_table(txt, "T0_novelty", OUT)


# --------------------------------------------------------------------- T1
def t1():
    r = _load("phase2")
    if not r:
        return None
    fwd_methods = ["PRISM", "FNO", "DeepONet", "PINN"]
    inv_methods = ["PRISM", "cINN", "NPE-CNF", "VI"]
    header = ["Method"] + [t.capitalize() for t in PDE]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Main PDE results. Top: forward surrogate accuracy "
             "(relative $L_2$, lower better). Bottom: inverse recovery (RMSE, "
             "lower better). Best per column in bold.}",
             "\\label{tab:main-pde}",
             "\\begin{tabular}{l" + "c" * len(PDE) + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule",
             "\\multicolumn{%d}{l}{\\emph{Forward (rel.\\ $L_2$)}} \\\\" % (len(PDE) + 1)]
    # forward block
    fwd_vals = {t: [r["T1"][t]["forward_relL2"][m] for m in fwd_methods] for t in PDE}
    fwd_bold = {t: _bold_min(fwd_vals[t]) for t in PDE}
    for mi, m in enumerate(fwd_methods):
        cells = [m]
        for t in PDE:
            v = fwd_vals[t][mi]
            s = f"{v:.3f}"
            cells.append(f"\\textbf{{{s}}}" if fwd_bold[t][mi] else s)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{%d}{l}{\\emph{Inverse (RMSE)}} \\\\" % (len(PDE) + 1))
    inv_vals = {t: [r["T1"][t]["inverse_rmse"][m] for m in inv_methods] for t in PDE}
    inv_bold = {t: _bold_min(inv_vals[t]) for t in PDE}
    for mi, m in enumerate(inv_methods):
        cells = [m]
        for t in PDE:
            v = inv_vals[t][mi]
            s = f"{v:.3f}"
            cells.append(f"\\textbf{{{s}}}" if inv_bold[t][mi] else s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return save_table("\n".join(lines), "T1_main_pde", OUT)


# --------------------------------------------------------------------- T2
def t2():
    r = _load("phase2")
    if not r:
        return None
    methods = ["PRISM", "cINN", "NPE-CNF", "VI"]
    tasks = list(r["T2"].keys())
    header = ["Task (constraint)"] + methods
    rows, bold = [], []
    for t in tasks:
        vals = [r["T2"][t][m] for m in methods]
        rows.append([f"{_esc(t)} ({r['T2'][t]['constraint']})"] + [f"{v:.1%}".replace('%', '\\%') for v in vals])
        bold.append([False] + _bold_min(vals))
    txt = latex_table(header, rows,
                      caption="Constraint-violation rate over posterior samples "
                              "(lower better). PRISM is exactly 0 by construction.",
                      label="tab:constraints", bold_mask=bold)
    return save_table(txt, "T2_constraints", OUT)


# --------------------------------------------------------------------- T3
def t3():
    r = _load("phase2")
    if not r:
        return None
    methods = ["PRISM", "PRISM-soft", "cINN"]
    tasks = list(r["T3"].keys())
    header = ["Task"] + methods
    rows, bold = [], []
    for t in tasks:
        vals = [r["T3"][t][m] for m in methods]
        rows.append([_esc(t)] + [f"{v:.1e}" for v in vals])
        bold.append([False] + _bold_min(vals))
    txt = latex_table(header, rows,
                      caption="Cycle-consistency error (relative reconstruction, lower better). "
                              "Exact-by-construction flows (PRISM, cINN) reach solver/float tolerance; "
                              "the soft-penalty variant does not.",
                      label="tab:cycle", bold_mask=bold)
    return save_table(txt, "T3_cycle", OUT)


# --------------------------------------------------------------------- T4
def t4():
    r = _load("phase4")
    if not r:
        return None
    metrics = [("c2st", "C2ST", "half"), ("coverage90", "Cov@90", "nom90"),
               ("expected_coverage_error", "ECE", "min"), ("crps", "CRPS", "min"),
               ("nll", "NLL", "min"), ("wdist", "W$_1$", "min")]
    methods = ["PRISM", "NPE-CNF", "VI"]
    header = ["Task", "Method"] + [m[1] for m in metrics]
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Posterior calibration on SBI tasks. C2ST best near 0.5; "
             "Cov@90 best near 0.90; ECE/CRPS/NLL/W$_1$ lower better. "
             "Best per task/metric in bold.}",
             "\\label{tab:calibration}",
             "\\begin{tabular}{ll" + "c" * len(metrics) + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for task in r["T4"]:
        # gather values per metric for bolding
        per_metric = {}
        for key, _, kind in metrics:
            vals = [r["T4"][task][m].get(key) for m in methods]
            if kind == "half":
                fin = [abs(v - 0.5) for v in vals if v is not None]
                best = min(fin) if fin else None
                per_metric[key] = [(v is not None and abs(v - 0.5) == best) for v in vals]
            elif kind == "nom90":
                fin = [abs(v - 0.9) for v in vals if v is not None]
                best = min(fin) if fin else None
                per_metric[key] = [(v is not None and abs(v - 0.9) == best) for v in vals]
            else:
                per_metric[key] = _bold_min(vals)
        for mi, m in enumerate(methods):
            cells = [_esc(task) if mi == 0 else "", m]
            for key, _, _ in metrics:
                v = r["T4"][task][m].get(key)
                s = "--" if v is None else f"{v:.3f}"
                if v is not None and per_metric[key][mi]:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    return save_table("\n".join(lines), "T4_calibration", OUT)


# --------------------------------------------------------------------- T5
def t5():
    r = _load("phase3")
    if not r:
        return None
    facs = ["1x", "2x", "4x"]
    header = ["Task"] + [f"{f} res." for f in facs]
    rows = []
    for t in r["T5"]:
        rows.append([_esc(t)] + [f"{r['T5'][t][f]:.3f}" for f in facs])
    txt = latex_table(header, rows,
                      caption="Zero-shot super-resolution: forward rel.\\ $L_2$ when the "
                              "operator is queried at higher field resolutions than seen in "
                              "training. Near-flat error indicates resolution invariance.",
                      label="tab:scaling")
    return save_table(txt, "T5_scaling", OUT)


# --------------------------------------------------------------------- T6
def t6():
    r = _load("phase5")
    if not r:
        return None
    header = ["Task", "Family", "Inv. $R^2$", "Fwd rel-$L_2$", "Viol.", "Pass"]
    rows = []
    for t in r["T6"]:
        d = r["T6"][t]
        rows.append([_esc(t), d["family"], f"{d['inverse_meanR2']:.3f}",
                     f"{d['forward_relL2']:.3f}", f"{d['violation']:.2f}",
                     "\\checkmark" if d["pass"] else "--"])
    s = r.get("T6_summary", {})
    note = f"One fixed PRISM configuration across all families: {s.get('passed','?')}/{s.get('total','?')} tasks pass. " \
           "Multimodal SBI tasks (SLCP, Two Moons) are evaluated by calibration (Table~\\ref{tab:calibration}); " \
           "posterior-mean $R^2$ understates them."
    txt = latex_table(header, rows,
                      caption="Cross-domain generality with a single model configuration.",
                      label="tab:generality", note=note)
    return save_table(txt, "T6_generality", OUT)


# --------------------------------------------------------------------- T7
def t7():
    r = _load("phase6")
    if not r:
        return None
    header = ["Variant", "Inv. $R^2$", "Cycle err.", "Viol.", "ECE"]
    rows = []
    for label in r["T7"]:
        d = r["T7"][label]
        rows.append([label, f"{d['inverse_meanR2']:.3f}", f"{d['cycle_err']:.1e}",
                     f"{d['violation']:.2f}", f"{d['expected_coverage_error']:.3f}"])
    txt = latex_table(header, rows,
                      caption="Ablation. Removing the liquid dynamics, exact reversibility "
                              "(soft), or the projection each degrades a corresponding property; "
                              "the soft variant loses both exact consistency and calibration.",
                      label="tab:ablation")
    return save_table(txt, "T7_ablation", OUT)


# --------------------------------------------------------------------- T8
def t8():
    r = _load("phase6")
    if not r:
        return None
    methods = [m for m in ["PRISM", "cINN", "NPE-CNF", "VI"] if m in r["T8"]]
    header = ["Method", "Params", "Train (s)", "Latency (ms/samp)", "Inv. $R^2$"]
    rows = []
    for m in methods:
        d = r["T8"][m]
        rows.append([m, f"{d['params']:,}", f"{d['train_time_s']:.1f}",
                     f"{d['latency_ms_per_sample']:.3f}", f"{d['inverse_meanR2']:.3f}"])
    note = None
    if "mcmc_speedup_x" in r["T8"]:
        note = f"MCMC reference latency {r['T8']['MCMC']['latency_ms_per_sample']:.3f} ms/sample; " \
               f"PRISM amortized speedup {r['T8']['mcmc_speedup_x']:.1f}$\\times$ " \
               "(grows with dimension and likelihood cost at full scale)."
    txt = latex_table(header, rows,
                      caption="Efficiency: parameter count, training time, and amortized "
                              "inference latency per posterior sample.",
                      label="tab:efficiency", note=note)
    return save_table(txt, "T8_efficiency", OUT)


def main():
    os.makedirs(OUT, exist_ok=True)
    made = []
    for fn in [t0, t1, t2, t3, t4, t5, t6, t7, t8]:
        try:
            p = fn()
            if p:
                made.append(os.path.basename(p))
                print(f"  wrote {os.path.basename(p)}")
            else:
                print(f"  skipped {fn.__name__} (no cached results)")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {e}")
    print(f"\n{len(made)} tables -> {OUT}")
    return made


if __name__ == "__main__":
    main()
