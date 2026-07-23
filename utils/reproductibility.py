"""
reproducibility.py
------------------
Utilities for ensuring reproducible training runs.
"""


import os
import random
import numpy as np
import torch



def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Set all random seeds for reproducibility.


    Parameters
    ----------
    seed          : Random seed
    deterministic : If True, set CUDNN to deterministic mode (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True can speed up training when input sizes are constant
        torch.backends.cudnn.benchmark = True