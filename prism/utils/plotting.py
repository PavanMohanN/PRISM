"""Publication plotting style + save helpers (vector PDF, colour-blind-safe).

Import `set_style()` once at the top of any figure script. Use `save_fig()`
to write deterministic, tight-bbox PDFs into paper_assets/.
"""
from __future__ import annotations

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# Colour-blind-safe qualitative palette (Wong, 2011). PRISM is always index 0.
PALETTE = [
    "#0072B2",  # blue   -> PRISM
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#999999",  # grey
]

METHOD_COLORS = {
    "PRISM": PALETTE[0], "PRISM-soft": PALETTE[1],
    "cINN": PALETTE[2], "inv-FNO": PALETTE[3], "FNO": PALETTE[4],
    "NPE-CNF": PALETTE[5], "DeepONet": PALETTE[6], "PINN": PALETTE[7],
    "amortized-VI": PALETTE[3], "MCMC": "#000000",
}


def set_style(fontsize: int = 9) -> None:
    """Apply a consistent ICLR/AISTATS-friendly style to all figures."""
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,        # editable text in the PDF (no type-3)
        "ps.fonttype": 42,
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "legend.fontsize": fontsize - 1,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


def color_for(method: str, idx: int = 0) -> str:
    return METHOD_COLORS.get(method, PALETTE[idx % len(PALETTE)])


def save_fig(fig, name: str, outdir: str = "paper_assets") -> str:
    """Save a figure as PDF (vector) into outdir; returns the path."""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name if name.endswith(".pdf") else name + ".pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
