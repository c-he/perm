# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Tuple, Union

import click
import numpy as np
import torch
import torch.nn.functional as F

import dnnlib
import legacy
from hair import HairRoots, save_hair
from models.neural_texture import RawNeuralTexture
from torch_utils import misc
from utils.image import write_texture
from utils.misc import copy2cpu as c2c

# ----------------------------------------------------------------------------


def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

# ----------------------------------------------------------------------------


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple):
        return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=8, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    head_mesh: str,
    scalp_bounds: List[float],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = RawNeuralTexture(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    exp_name = network_pkl.split(os.path.sep)[-2]
    network_pkl = os.path.basename(network_pkl)
    outdir = os.path.join(outdir, f"{exp_name.split('-')[0]}-{network_pkl[:-4]}")
    os.makedirs(outdir, exist_ok=True)

    hair_roots = HairRoots(head_mesh=head_mesh, scalp_bounds=scalp_bounds)
    u, v = torch.meshgrid(torch.linspace(0, 1, steps=256), torch.linspace(0, 1, steps=256), indexing='ij')
    uv = torch.dstack((u, v)).permute(2, 1, 0)  # (2, H, W)
    uv_guide = F.interpolate(uv.unsqueeze(0), size=(G.img_resolution, G.img_resolution), mode='nearest')[0]
    uv_guide = uv_guide.permute(1, 2, 0).reshape(-1, 2)
    uv_guide = hair_roots.rescale(uv_guide, inverse=True)
    guide_roots = hair_roots.spherical_to_cartesian(uv_guide).unsqueeze(0)
    guide_roots = guide_roots.to(device)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        out = G(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')
        image = out['image']
        image_mask = out['image_mask']
        guide_strands = G.sample(image)
        guide_strands.position = guide_strands.position + guide_roots.unsqueeze(2)
        write_texture(os.path.join(outdir, f'seed{seed:04d}.png'), image[0].permute(1, 2, 0), alpha=image_mask[0].permute(1, 2, 0))
        save_hair(os.path.join(outdir, f'seed{seed:04d}.obj'), c2c(guide_strands.position[0]))
        np.savez(os.path.join(outdir, f'seed{seed:04d}.npz'), texture=c2c(image[0]), mask=c2c(image_mask[0]), roots=c2c(uv_guide))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
