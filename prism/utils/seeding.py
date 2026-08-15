"""Global seeding for reproducibility across python / numpy / torch / cuda."""
from __future__ import annotations

import os
import random
import contextlib

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs. Call once at the start of every experiment script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def temp_seed(seed: int):
    """Temporarily set seeds inside a `with` block, then restore RNG state."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state() if _HAS_TORCH else None
    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if _HAS_TORCH and torch_state is not None:
            torch.set_rng_state(torch_state)
