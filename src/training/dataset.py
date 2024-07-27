import glob
import os

import numpy as np
from torch.utils.data import Dataset

from utils.misc import load_tensor_dict


class ParametricHairDataset(Dataset):
    def __init__(
        self,
        hair_path,          # Path to hair .npz data.
        tex_path=None,      # Path to .npz neural textures.
        resolution=None,    # Ensure specific resolution, None = highest available.
        hair_prop=[],       # Data properties to load from .npz files for hair. Empty means loading all properties.
        max_size=None,      # Artificially limit the size of the dataset. None = no limit.
    ):
        self._name = hair_path.split(os.path.sep)[-2]  # Name of the dataset.
        self._hair_path = hair_path
        self._tex_path = tex_path
        self._all_fnames = [os.path.basename(f) for f in glob.glob(os.path.join(hair_path, "*.npz"))]
        self._all_fnames = sorted(self._all_fnames)

        self._hair_prop = hair_prop
        if len(hair_prop) == 0:
            hair_prop = list(self._load_hair_data(self._all_fnames[0]).keys())
            self._hair_prop = hair_prop
        if tex_path is not None:
            raw_shape = [len(self._all_fnames)] + list(self._load_texture(self._all_fnames[0]).shape)
            if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
                raise IOError('Image files do not match the specified resolution')
            self._raw_shape = raw_shape

        # Apply max_size.
        self._raw_idx = np.arange(len(self._all_fnames), dtype=np.int64)
        if (max_size is not None) and len(self._all_fnames) > max_size:
            self._raw_idx = self._raw_idx[:max_size]

    def _load_hair_data(self, fname):
        return load_tensor_dict(os.path.join(self._hair_path, fname), keys=self._hair_prop)

    def _load_texture(self, fname):
        if self._tex_path is not None:
            return load_tensor_dict(os.path.join(self._tex_path, fname), keys=['data'])['data']
        return None

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        fname = self._all_fnames[self._raw_idx[idx]]
        hair = self._load_hair_data(fname)
        tex = self._load_texture(fname)

        return hair, tex, idx

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def hair_prop(self):
        return self._hair_prop


class NeuralTextureDataset(Dataset):
    def __init__(
        self,
        path,               # Path to .npz neural textures.
        max_size=None,      # Artificially limit the size of the dataset. None = no limit.
        **kwargs
    ):
        self._name = 'hair-neural-textures'  # Name of the dataset.
        self._path = path
        self._img_path = os.path.join(path, 'high-res')
        self._raw_path = os.path.join(path, 'low-res')
        self._all_fnames = [os.path.basename(f) for f in glob.glob(os.path.join(self._img_path, '*.npz'))]
        self._all_fnames = sorted(self._all_fnames)

        self._img_shape = [len(self._all_fnames)] + list(self._load_texture(os.path.join(self._img_path, self._all_fnames[0]))['texture'].shape)
        self._raw_shape = [len(self._all_fnames)] + list(self._load_texture(os.path.join(self._raw_path, self._all_fnames[0]))['texture'].shape)

        # Apply max_size.
        self._raw_idx = np.arange(len(self._all_fnames), dtype=np.int64)
        if (max_size is not None) and len(self._all_fnames) > max_size:
            self._raw_idx = self._raw_idx[:max_size]

    def _load_texture(self, fname):
        return load_tensor_dict(fname)

    def get_raw_texture(self, idx):
        fname = self._all_fnames[self._raw_idx[idx]]
        return self._load_texture(os.path.join(self._raw_path, fname))['texture']

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        fname = self._all_fnames[self._raw_idx[idx]]
        img = self._load_texture(os.path.join(self._img_path, fname))
        raw = self._load_texture(os.path.join(self._raw_path, fname))

        return img['texture'], raw['texture'], img['mask'], raw['mask']

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def img_shape(self):
        return list(self._img_shape[1:])

    @property
    def raw_shape(self):
        return list(self._raw_shape[1:])

    @property
    def img_channels(self):
        assert len(self.img_shape) == 3  # CHW
        return self.img_shape[0]

    @property
    def raw_channels(self):
        assert len(self.raw_shape) == 3  # CHW
        return self.raw_shape[0]

    @property
    def img_resolution(self):
        assert len(self.img_shape) == 3  # CHW
        assert self.img_shape[1] == self.img_shape[2]
        return self.img_shape[1]

    @property
    def raw_resolution(self):
        assert len(self.raw_shape) == 3  # CHW
        assert self.raw_shape[1] == self.raw_shape[2]
        return self.raw_shape[1]
