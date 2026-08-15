"""PRISM Phase 3 -- shared variant construction (repo hooks in one place).

Every Phase 3 experiment toggles PRISM ablation flags. The exact flag names live
in your PRISM estimator; adjust the mapping in FLAG_ALIASES once and all four
scripts follow. Flags used:

    liquid        -- liquid time-constant field (True) vs static neural ODE (False)
    cond_base     -- observation-conditioned base distribution
    cond_velocity -- observation-conditioned velocity field g(v,t,y)
    use_projection-- apply the support transform / projection

Defaults (full PRISM): liquid=True, cond_base=True, cond_velocity=False,
use_projection=True.
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# map our canonical flag -> your estimator's kwarg name, if different
FLAG_ALIASES = {
    "liquid": "liquid",
    "cond_base": "cond_base",
    "cond_velocity": "cond_velocity",
    "use_projection": "use_projection",
}


def make_variant(task, cfg, **flags):
    """Construct a PRISM variant with the given ablation flags."""
    kwargs = {FLAG_ALIASES.get(k, k): v for k, v in flags.items()}
    try:
        from experiments.api import make_method
        return make_method("PRISM", task, cfg, **kwargs)
    except TypeError:
        # make_method doesn't accept flags -> construct estimator directly
        from prism.models.prism import PRISM
        base = dict(cfg.get("prism", {})) if hasattr(cfg, "get") else {}
        base.update(kwargs)
        return PRISM(**base)


def count_params(estimator) -> int:
    """Total trainable parameters, across whatever torch modules the estimator holds."""
    import torch
    total = 0
    seen = set()
    for attr in vars(estimator).values():
        if isinstance(attr, torch.nn.Module) and id(attr) not in seen:
            seen.add(id(attr))
            total += sum(p.numel() for p in attr.parameters() if p.requires_grad)
    return total
