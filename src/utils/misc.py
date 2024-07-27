import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

EPSILON = 1e-7


class Struct(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def copy2cpu(data: Union[torch.Tensor, Dict]) -> Union[np.ndarray, Dict]:
    if isinstance(data, dict):
        return {k: v.detach().cpu().numpy() for k, v in data.items()}
    return data.detach().cpu().numpy()


def filename(f: str) -> str:
    return os.path.splitext(os.path.basename(f))[0]


def load_tensor_dict(path: str, keys: Optional[List[str]] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    data = np.load(path)
    if keys is None or len(keys) == 0:
        keys = list(data.keys())

    result = dict()
    for key in keys:
        if device is None:
            result[key] = torch.tensor(data[key], dtype=torch.float32)
        else:
            result[key] = torch.tensor(data[key], dtype=torch.float32, device=device)

    return result


def flatten_list(l: List[Any]) -> List[Any]:
    return [item for sublist in l for item in sublist]
