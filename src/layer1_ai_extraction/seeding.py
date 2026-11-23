"""Global seed fixation (thesis 3.4.1)."""

from __future__ import annotations

import os
import random

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed `random`, NumPy, PyTorch (CPU and CUDA) and `transformers`.

    `PYTHONHASHSEED` is written to the environment, but Python fixes string hashing at
    start-up, so the value only affects child processes. Nothing in Layer 1 depends on
    `hash()`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass
    return seed
