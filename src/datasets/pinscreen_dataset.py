import torch
from torch.utils.data import Dataset

from datasets.pinscreen import load_neural_textures, load_pinscreen_data


class PinscreenHair(Dataset):
    def __init__(self, **kwargs):
        data_keys = kwargs.get('data_keys', None)
        range = kwargs.get('range', None)
        self.ds = load_pinscreen_data(kwargs['hair_data'], keys=data_keys, range=range, num_workers=1, reduction='none')

        neural_texture = kwargs.get('neural_texture', None)
        if neural_texture is not None:
            self.ds.update({'texture': load_neural_textures(neural_texture, range=range, reduction='stack')})

    def __len__(self):
        return len(list(self.ds.values())[0])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}
        data['idx'] = idx

        return data


class PinscreenStrands(Dataset):
    def __init__(self, **kwargs):
        self.ds = torch.load(kwargs['hair_data'])
        for k, v in self.ds.items():
            print(f'{k}: {v.shape}')

    def __len__(self):
        return len(list(self.ds.values())[0])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys()}

        return data
