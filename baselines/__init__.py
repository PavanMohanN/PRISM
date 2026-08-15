"""Baseline methods compared against PRISM.

Posterior/inverse methods (fit / invert / predict_posterior):
    CINN, NPECNF, AmortizedVI, MCMCReference
Forward-operator surrogates (fit(fields, Y) / predict(fields)):
    FNO, DeepONet, PINN
"""
from .cinn import CINN
from .npe_cnf import NPECNF
from .amortized_vi import AmortizedVI
from .mcmc_reference import MCMCReference
from .fno import FNO
from .deeponet import DeepONet
from .pinn import PINN

__all__ = ["CINN", "NPECNF", "AmortizedVI", "MCMCReference",
           "FNO", "DeepONet", "PINN"]
