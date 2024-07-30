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

import glob
import os
import re
from typing import List, Optional, Union

import click
import numpy as np
import torch
from tqdm import tqdm

import dnnlib
import legacy
from hair import HairRoots, save_hair
from models.neural_texture import ResNeuralTexture
from torch_utils import misc
from utils.image import write_texture
from utils.misc import copy2cpu as c2c
from utils.misc import filename, load_tensor_dict
from utils.visualize import plot_weight_std

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
@click.option('--source', help='Filename for hair neural textures', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    source: str,
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
        G = legacy.load_network_pkl(f)['G'].to(device)  # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = ResNeuralTexture(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    # exp_name = network_pkl.split(os.path.sep)[-2]
    # network_pkl = os.path.basename(network_pkl)
    # outdir = os.path.join(outdir, f"{exp_name.split('-')[0]}-{network_pkl[:-4]}")
    os.makedirs(outdir, exist_ok=True)
    hair_roots = HairRoots(head_mesh=head_mesh, scalp_bounds=scalp_bounds)

    fname = filename(source)
    data = load_tensor_dict(source, device=device)
    low_rank_coeff = data['texture'][:G.raw_channels]
    roots = data['roots']
    coords = hair_roots.rescale(roots[..., :2])

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        ws = torch.from_numpy(np.random.RandomState(seed).randn(1, 1, G.w_dim)).to(device)
        ws = ws.repeat(1, G.num_ws, 1)
        high_rank_coeff = G.synthesis(ws=ws, noise_mode='const')['image']
        write_texture(os.path.join(outdir, f'seed{seed:04d}.png'), high_rank_coeff[0].permute(1, 2, 0))
        image = torch.cat([low_rank_coeff.unsqueeze(0), high_rank_coeff], dim=1)
        strands = G.sample(image, coords.unsqueeze(0), mode='nearest')[0]
        strands.position = strands.position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
        save_hair(os.path.join(outdir, f'{fname}_seed{seed:04d}.obj'), c2c(strands.position))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
