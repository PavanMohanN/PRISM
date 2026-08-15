"""Headline claim #2: hard constraints satisfied by construction.

Every decoded parameter must be feasible regardless of the latent draw, so the
violation rate is exactly 0 for all supported constraint types.
"""
import numpy as np
import torch
from prism.constraints.projections import get_constraint


def test_positive_never_negative():
    c = get_constraint("positive")
    x = c.forward(torch.randn(5000, 6, dtype=torch.float64))
    assert (x > 0).all()
    assert float((c.violation(x) > 1e-6).float().mean()) == 0.0


def test_box_within_bounds():
    c = get_constraint(("box", {"lo": -1.0, "hi": 2.0}))
    x = c.forward(torch.randn(5000, 4, dtype=torch.float64))
    assert (x >= -1.0).all() and (x <= 2.0).all()


def test_simplex_sums_to_one_and_nonneg():
    c = get_constraint("simplex")
    x = c.forward(torch.randn(5000, 5, dtype=torch.float64))
    assert (x >= 0).all()
    assert torch.allclose(x.sum(dim=1), torch.ones(5000, dtype=torch.float64), atol=1e-6)


def test_monotone_nondecreasing():
    c = get_constraint("monotone")
    x = c.forward(torch.randn(5000, 8, dtype=torch.float64))
    assert (x[:, 1:] - x[:, :-1] >= -1e-6).all()


def test_inverse_roundtrip_positive():
    c = get_constraint("positive")
    x = torch.rand(100, 3, dtype=torch.float64) + 0.1
    assert torch.allclose(c.forward(c.inverse(x)), x, atol=1e-4)
