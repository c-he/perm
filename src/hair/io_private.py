try:
    import stylist
except:
    raise ModuleNotFoundError("Module Stylist is required for hair data I/O but not found in the environment. "
                              "To install it, please check: https://git.corp.adobe.com/research-hair-dataset/stylist or https://git.corp.adobe.com/chhe/stylist")
import os
from typing import Optional

import numpy as np


def load_hair(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1]
    assert ext in ['.data', '.abc'], "only support loading hair data in .data and .abc formats"

    h = stylist.hair()
    h.load(path)

    return np.array(h.strands)


def save_hair(path: str, data: np.ndarray, color: Optional[np.ndarray] = None):
    ext = os.path.splitext(path)[1]
    assert ext in ['.data', '.abc', '.obj', '.ply'], "only support saving hair data in .data, .abc, .obj and .ply formats"

    h = stylist.hair()
    h.strands = data
    if ext == '.ply':
        h.save_ply_binary_with_colors(path, color) if color is not None else h.save_ply_binary_with_random_strand_colors(path)
    else:
        h.save(path)
