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
from typing import List, Optional

import click
import torch
import torch.nn.functional as F
from tqdm import tqdm

import dnnlib
import legacy
from hair import HairRoots, save_hair
from models.neural_texture import NeuralTextureSuperRes
from torch_utils import misc
from utils.image import write_texture
from utils.misc import copy2cpu as c2c
from utils.misc import filename, load_tensor_dict
from utils.visualize import plot_texture_error

# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


# ----------------------------------------------------------------------------


@click.command()
@click.option('--sr_mode', help='Super resolution mode', metavar='STR', type=click.Choice(['nearest', 'bilinear', 'neural']), required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--source', help='Directory for hair neural textures', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--max_images', help='Output only up to `max-images` images', type=int, default=None)
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    sr_mode: str,
    network_pkl: str,
    source: str,
    outdir: str,
    max_images: Optional[int],
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
        G_new = NeuralTextureSuperRes(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    os.makedirs(outdir, exist_ok=True)
    hair_roots = HairRoots(head_mesh=head_mesh, scalp_bounds=scalp_bounds)

    # image_files = sorted(glob.glob(os.path.join(source, 'low-res', '*.npz')))
    image_files = sorted(set(glob.glob(os.path.join(source, 'low-res', '*.npz'))) - set(glob.glob(os.path.join(source, 'low-res', '*_*.npz'))))
    if max_images is not None:
        image_files = image_files[:max_images]
    for f in tqdm(image_files):
        fname = filename(f)
        data = load_tensor_dict(f, device=device)
        raw_image = data['texture'].unsqueeze(0)
        raw_mask = data['mask'].unsqueeze(0)
        if sr_mode == 'nearest':
            image = F.interpolate(raw_image, (G.img_resolution, G.img_resolution), mode='nearest')
        elif sr_mode == 'bilinear':
            image = F.interpolate(raw_image, (G.img_resolution, G.img_resolution), mode='bilinear', align_corners=False, antialias=True)
        else:
            img = {'image_raw': raw_image, 'image_mask': raw_mask}
            out = G(img)
            image = out['image']
        write_texture(os.path.join(outdir, f'{fname}.png'), image[0].permute(1, 2, 0))
        gt_data = load_tensor_dict(os.path.join(source, 'high-res', f'{fname}.npz'), device=device)
        gt_image = gt_data['texture'][:G.img_channels]
        write_texture(os.path.join(outdir, f'{fname}_gt.png'), gt_image.permute(1, 2, 0))
        error = torch.norm(image - gt_image, dim=1)
        plot_texture_error(os.path.join(outdir, f'{fname}_err.png'), error[0])

        roots = gt_data['roots']
        coords = hair_roots.rescale(roots[..., :2])

        strands = G.sample(image, coords.unsqueeze(0), mode='nearest')[0]
        strands.position = strands.position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
        save_hair(os.path.join(outdir, f'{fname}.data'), c2c(strands.position))
        gt_strands = G.sample(gt_image.unsqueeze(0), coords.unsqueeze(0), mode='nearest')[0]
        gt_strands.position = gt_strands.position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
        save_hair(os.path.join(outdir, f'{fname}_gt.data'), c2c(gt_strands.position))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
