"""PRISM utilities: seeding, metrics, plotting, latex tables."""
from .seeding import set_seed, temp_seed
from .metrics import (
    rmse, mae, relative_l2, r2_score, psnr, ssim, aggregate,
)
from .plotting import set_style, save_fig, color_for, PALETTE, METHOD_COLORS
from .latex import latex_table, save_table, fmt_pm

__all__ = [
    "set_seed", "temp_seed",
    "rmse", "mae", "relative_l2", "r2_score", "psnr", "ssim", "aggregate",
    "set_style", "save_fig", "color_for", "PALETTE", "METHOD_COLORS",
    "latex_table", "save_table", "fmt_pm",
]
