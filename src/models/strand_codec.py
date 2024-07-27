from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils.blend_shape import blend, project


class StrandCodec(nn.Module):
    SAMPLES_PER_STRAND = 100

    def __init__(self, model_path: str, num_coeff: Optional[int] = None, fft: bool = True):
        super().__init__()

        data = np.load(model_path)
        self.register_buffer('mean_shape', torch.tensor(data['mean_shape'], dtype=torch.float32))

        if num_coeff is not None:
            num_coeff = min(num_coeff, data['blend_shapes'].shape[0])
        else:
            num_coeff = data['blend_shapes'].shape[0]
        self.register_buffer('blend_shapes', torch.tensor(data['blend_shapes'][:num_coeff], dtype=torch.float32))

        self.num_coeff = num_coeff
        self.fft = fft

    def encode(self, x):
        if self.fft:
            fourier = torch.fft.rfft(x, n=self.SAMPLES_PER_STRAND - 1, dim=-2, norm='ortho')
            x = torch.cat((fourier.real, fourier.imag), dim=-1)
        return project(x - self.mean_shape, self.blend_shapes)

    def decode(self, coeff):
        x = self.mean_shape + blend(coeff, self.blend_shapes)
        if self.fft:
            x = torch.fft.irfft(torch.complex(x[..., :3], x[..., 3:]), n=self.SAMPLES_PER_STRAND - 1, dim=-2, norm='ortho')
        return x

    def forward(self, x):
        coeff = self.encode(x)
        return self.decode(coeff)
