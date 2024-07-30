# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import glob
import os
from time import perf_counter
from typing import List

import click
import imageio
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy
from hair import HairRoots, save_hair
from hair.loss import StrandGeometricLoss
from models.neural_texture import RawNeuralTexture
from torch_utils import misc
from utils.metric import curvature
from utils.misc import copy2cpu as c2c
from utils.misc import filename, load_tensor_dict

# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')


# ----------------------------------------------------------------------------

def project(
    G,
    target_image: torch.Tensor,
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    optimize_noise=False,
    verbose=False,
    device: torch.device
):
    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device))     # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)                 # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    start_w = np.repeat(w_avg, G.backbone.mapping.num_ws, axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    if optimize_noise:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True
    else:
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    criterion = StrandGeometricLoss()
    target_hair = G.sample(target_image)

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        synth_image = G.synthesis(ws, noise_mode='const')['image']
        synth_hair = G.sample(synth_image)
        strand_loss = criterion(synth_hair, target_hair)
        tex_loss = F.l1_loss(target_image, synth_image)

        # Noise regularization.
        reg_loss = 0.0
        if optimize_noise:
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)
        loss = strand_loss['pos'] + strand_loss['rot'] + strand_loss['cur'] + tex_loss + regularize_noise_weight * reg_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f"step {step+1:>4d}/{num_steps}: pos {strand_loss['pos']:<4.2f} rot {strand_loss['rot']:<4.2f} cur {strand_loss['cur']:<4.2f} tex {tex_loss:<4.2f} noise_reg {reg_loss:<4.2f}")

        # Save projected W for each optimization step.)
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        if optimize_noise:
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

    if w_out.shape[1] == 1:
        w_out = w_out.repeat([1, G.backbone.mapping.num_ws, 1])

    return w_out

# ----------------------------------------------------------------------------


def image_to_rgb(image, rgb_size=None):
    lo, high = image.min(), image.max()
    rgb = ((image - lo) / (high - lo))[:3]  # scale to [0, 1]
    if (rgb_size is not None) and (rgb.shape[1] < rgb_size or rgb.shape[2] < rgb_size):
        rgb = F.interpolate(rgb.unsqueeze(0), size=(rgb_size, rgb_size), mode='nearest')[0]
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return c2c(rgb.permute(1, 2, 0))

# ----------------------------------------------------------------------------


def run_projection(G, target, outdir, hair_roots, num_steps, save_video, device):
    # Load target data.
    target_data = load_tensor_dict(target, device=device)
    target_image = target_data['texture'].unsqueeze(0)
    target_rgb = image_to_rgb(target_image[0], rgb_size=256)
    roots = target_data['roots']

    # Optimize projection.
    print(f'Start optimization...')
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target_image=target_image,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print(f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    fname = filename(target)
    outdir = os.path.join(outdir, fname)
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/{fname}_proj.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps[::3]:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')['image']
            video.append_data(np.concatenate([target_rgb, image_to_rgb(synth_image[0], rgb_size=256)], axis=1))
        video.close()

    # Save final projected frame and W vector.
    projected_w = projected_w_steps[-1]
    synth = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = synth['image']
    synth_mask = synth['image_mask']
    synth_rgb = image_to_rgb(synth_image[0], rgb_size=256)
    PIL.Image.fromarray(target_rgb, 'RGB').save(f'{outdir}/{fname}_gt.png')
    PIL.Image.fromarray(synth_rgb, 'RGB').save(f'{outdir}/{fname}_proj.png')
    np.savez(f'{outdir}/{fname}.npz', texture=c2c(synth_image[0]), mask=c2c(synth_mask[0]), roots=c2c(roots), w=c2c(projected_w))
    target_hair = G.sample(target_image)[0]
    target_hair.position = target_hair.position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
    save_hair(f'{outdir}/{fname}_gt.data', c2c(target_hair.position))
    synth_hair = G.sample(synth_image)[0]
    synth_hair.position = synth_hair.position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
    save_hair(f'{outdir}/{fname}_proj.data', c2c(synth_hair.position))

    pos_diff = torch.norm(target_hair.position - synth_hair.position, dim=-1)
    cur_diff = (curvature(target_hair.position) - curvature(synth_hair.position)).abs()

    return c2c(pos_diff.mean()), c2c(cur_diff.mean())

# ----------------------------------------------------------------------------


@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', help='Target hair neural texture to project to', required=True, metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def main(
    network_pkl: str,
    target: str,
    outdir: str,
    head_mesh: str,
    scalp_bounds: List[float],
    save_video: bool,
    seed: int,
    num_steps: int,
    reload_modules: bool,
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(False).to(device)  # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = RawNeuralTexture(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G = G_new

    hair_roots = HairRoots(head_mesh=head_mesh, scalp_bounds=scalp_bounds)

    if os.path.isdir(target):
        target_files = sorted(glob.glob(os.path.join(target, '*.npz')))
        metrics = dict(pos_diff=[], cur_diff=[])
        for f in target_files:
            pos_diff, cur_diff = run_projection(G, f, outdir, hair_roots, num_steps, save_video, device)
            metrics['pos_diff'].append(pos_diff)
            metrics['cur_diff'].append(cur_diff)

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(os.path.join(outdir, f'recon_metrics.csv'))
    else:
        pos_diff, cur_diff = run_projection(G, target, outdir, hair_roots, num_steps, save_video, device)
        print(f'pos diff: {pos_diff}')
        print(f'cur diff: {cur_diff}')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
