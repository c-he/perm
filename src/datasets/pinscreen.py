import glob
import os
from itertools import repeat
from typing import List, Optional

import numpy as np
import torch
from torch.multiprocessing import Pool
from tqdm import tqdm


def load_single_pinscreen_data(data_fname: str, keys: Optional[List[str]] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    torch.set_num_threads(1)
    data = np.load(data_fname)
    if keys is None:
        keys = list(data.keys())

    result = dict()
    for key in keys:
        if device is None:
            result[key] = torch.tensor(data[key], dtype=torch.float32)
        else:
            result[key] = torch.tensor(data[key], dtype=torch.float32, device=device)

    return result


def load_pinscreen_data(root: str, keys: Optional[List[str]] = None, range: Optional[List[int]] = None, num_workers: int = -1, reduction: str = 'none') -> torch.Tensor:
    dataset = dict()
    data_fnames = sorted(glob.glob(os.path.join(root, '*.npz')))
    if range is not None:
        data_fnames = data_fnames[range[0]:range[1]]

    if num_workers > 1:  # threading loading data
        p = Pool(num_workers)
        try:
            # iterator = p.imap(_parallel_load_pinscreen_data, data_fnames)
            iterator = p.starmap(load_single_pinscreen_data, zip(data_fnames, repeat(keys)))
            for _ in tqdm(range(len(data_fnames)), desc='Loading data'):
                data = next(iterator)
                for k, v in data.items():
                    if k in dataset:
                        dataset[k].append(v)
                    else:
                        dataset[k] = [v]
        finally:
            p.close()
            p.join()
    else:
        for data_fname in tqdm(data_fnames, desc='Loading data'):
            data = load_single_pinscreen_data(data_fname, keys)
            for k, v in data.items():
                if k in dataset:
                    dataset[k].append(v)
                else:
                    dataset[k] = [v]

    if reduction == 'none':
        pass
    elif reduction == 'cat':
        for k, v in dataset.items():
            dataset[k] = torch.cat(v)
    elif reduction == 'stack':
        for k, v in dataset.items():
            dataset[k] = torch.stack(v)
    else:
        raise RuntimeError(f'unsupported reduction method for loaded Pinscreen data: {reduction}')

    return dataset


def load_neural_textures(root: str, range: Optional[List[int]] = None, reduction: str = 'none') -> torch.Tensor:
    dataset = []
    data_fnames = sorted(glob.glob(os.path.join(root, '*.npz')))
    if range is not None:
        data_fnames = data_fnames[range[0]:range[1]]

    for data_fname in tqdm(data_fnames, desc='Loading neural textures'):
        data = np.load(data_fname)
        dataset.append(torch.tensor(data['data'], dtype=torch.float32))

    if reduction == 'none':
        pass
    elif reduction == 'cat':
        dataset = torch.cat(dataset)
    elif reduction == 'stack':
        dataset = torch.stack(dataset)
    else:
        raise RuntimeError(f'unsupported reduction method for loaded neural textures: {reduction}')

    return dataset
