"""Minimal, dependency-free LaTeX (booktabs) table writer.

Produces \\begin{table} ... \\end{table} strings with \\toprule/\\midrule/
\\bottomrule, bolded best cells, and optional mean\\pmstd formatting.
"""
from __future__ import annotations

import os
from typing import Sequence


def fmt_pm(mean: float, std: float | None = None, prec: int = 3) -> str:
    if std is None:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f}$\\pm${std:.{prec}f}"


def latex_table(
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    caption: str,
    label: str,
    bold_mask: Sequence[Sequence[bool]] | None = None,
    col_align: str | None = None,
    note: str | None = None,
) -> str:
    """Build a booktabs table string. `rows` are pre-formatted strings."""
    ncol = len(header)
    align = col_align or ("l" + "c" * (ncol - 1))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for i, row in enumerate(rows):
        cells = []
        for j, c in enumerate(row):
            if bold_mask is not None and bold_mask[i][j]:
                c = f"\\textbf{{{c}}}"
            cells.append(c)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append(f"\\\\[2pt]\\footnotesize {note}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def save_table(text: str, name: str, outdir: str = "paper_assets") -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name if name.endswith(".tex") else name + ".tex")
    with open(path, "w") as f:
        f.write(text + "\n")
    return path
