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
from typing import List, Optional

import click
import imageio
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torch.nn.functional as F

from hair import save_hair
from hair.hair_models import Perm
from hair.loss import StrandGeometricLoss
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

def project_theta(
    hair_model,
    target_image: torch.Tensor,
    roots: torch.Tensor,
    *,
    num_steps=1000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    verbose=False,
    device: torch.device
):
    def logprint(*args):
        if verbose:
            print(*args)

    hair_model = copy.deepcopy(hair_model).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    theta_avg = hair_model.theta_avg()
    beta_avg = hair_model.beta_avg()

    theta_opt = theta_avg.detach().clone().requires_grad_(True)
    theta_out = torch.zeros([num_steps] + list(theta_opt.shape[1:]), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([theta_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    criterion = StrandGeometricLoss()
    coords = hair_model.hair_roots.cartesian_to_spherical(roots)[..., :2]
    coords = hair_model.hair_roots.rescale(coords)
    target_hair = hair_model.G_res.sample(target_image, coords, mode='nearest')
    target_hair.position = target_hair.position + roots.unsqueeze(2)

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        noise_scale = initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images.
        theta_noise = torch.randn_like(theta_opt) * noise_scale
        theta = theta_opt + theta_noise
        out = hair_model(roots=roots, theta=theta, beta=beta_avg)
        synth_image = out['image']
        synth_hair = out['strands']
        strand_loss = criterion(synth_hair, target_hair)
        tex_loss = F.l1_loss(target_image, synth_image)
        loss = strand_loss['pos'] + strand_loss['rot'] + strand_loss['cur'] + tex_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f"step {step+1:>4d}/{num_steps}: pos {strand_loss['pos']:<4.2f} rot {strand_loss['rot']:<4.2f} cur {strand_loss['cur']:<4.2f} tex {tex_loss:<4.2f}")

        # Save projected W for each optimization step.
        theta_out[step] = theta_opt.detach()[0]

    return theta_out

# ----------------------------------------------------------------------------


def project(
    hair_model,
    target_image: torch.Tensor,
    roots: torch.Tensor,
    start_theta: Optional[torch.Tensor],
    *,
    num_steps=1000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    verbose=False,
    device: torch.device
):
    def logprint(*args):
        if verbose:
            print(*args)

    hair_model = copy.deepcopy(hair_model).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    if start_theta is None:
        start_theta = hair_model.theta_avg()
    beta_avg = hair_model.beta_avg()

    theta_opt = start_theta.detach().clone().requires_grad_(True)
    beta_opt = beta_avg.detach().clone().requires_grad_(True)

    theta_out = torch.zeros([num_steps] + list(theta_opt.shape[1:]), dtype=torch.float32, device=device)
    beta_out = torch.zeros([num_steps] + list(beta_opt.shape[1:]), dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([theta_opt, beta_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
    criterion = StrandGeometricLoss()
    coords = hair_model.hair_roots.cartesian_to_spherical(roots)[..., :2]
    coords = hair_model.hair_roots.rescale(coords)
    target_hair = hair_model.G_res.sample(target_image, coords, mode='nearest')
    target_hair.position = target_hair.position + roots.unsqueeze(2)

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        noise_scale = initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images.
        theta_noise = torch.randn_like(theta_opt) * noise_scale
        beta_noise = torch.randn_like(beta_opt) * noise_scale
        theta = theta_opt + theta_noise
        beta = beta_opt + beta_noise
        out = hair_model(roots=roots, theta=theta, beta=beta)
        synth_image = out['image']
        synth_hair = out['strands']
        strand_loss = criterion(synth_hair, target_hair)
        tex_loss = F.l1_loss(target_image, synth_image)
        loss = strand_loss['pos'] + strand_loss['rot'] + strand_loss['cur'] + tex_loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f"step {step+1:>4d}/{num_steps}: pos {strand_loss['pos']:<4.2f} rot {strand_loss['rot']:<4.2f} cur {strand_loss['cur']:<4.2f} tex {tex_loss:<4.2f}")

        # Save projected W for each optimization step.
        theta_out[step] = theta_opt.detach()[0]
        beta_out[step] = beta_opt.detach()[0]

    return theta_out, beta_out

# ----------------------------------------------------------------------------


def image_to_rgb(image, rgb_size=None):
    lo, high = image.min(), image.max()
    rgb = ((image - lo) / (high - lo))[:3]  # scale to [0, 1]
    if (rgb_size is not None) and (rgb.shape[1] < rgb_size or rgb.shape[2] < rgb_size):
        rgb = F.interpolate(rgb.unsqueeze(0), size=(rgb_size, rgb_size), mode='nearest')[0]
    rgb = (rgb * 255).clamp(0, 255).to(torch.uint8)
    return c2c(rgb.permute(1, 2, 0))

# ----------------------------------------------------------------------------


def run_projection(hair_model, target, outdir, num_steps_warmup, num_steps, save_video, device):
    # Load target data.
    target_data = load_tensor_dict(target, device=device)
    target_image = target_data['texture']
    target_rgb = image_to_rgb(target_image, rgb_size=256)
    roots = target_data['roots']
    roots = hair_model.hair_roots.spherical_to_cartesian(target_data['roots'])

    # Optimize projection.
    print(f'Start optimization...')
    start_time = perf_counter()
    if num_steps_warmup > 0:  # warm-up
        warmup_steps = project_theta(
            hair_model=hair_model,
            target_image=target_image[None, ...],
            roots=roots[None, ...],
            num_steps=num_steps_warmup,
            device=device,
            verbose=True
        )
        start_theta = warmup_steps[-1:]
    else:
        warmup_steps = []
        start_theta = None
    theta_steps, beta_steps = project(
        hair_model=hair_model,
        target_image=target_image[None, ...],
        roots=roots[None, ...],
        start_theta=start_theta,
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
        os.makedirs(os.path.join(outdir, 'video'), exist_ok=True)
        video = imageio.get_writer(f'{outdir}/video/{fname}_proj.mp4', mode='I', fps=30, codec='libx264', bitrate='16M')
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for idx, theta in enumerate(warmup_steps[::20]):
            beta = hair_model.beta_avg()
            out = hair_model(roots=roots[None, ...], theta=theta[None, ...], beta=beta)
            synth_image = out['image']
            video.append_data(np.concatenate([target_rgb, image_to_rgb(synth_image[0], rgb_size=256)], axis=1))
            synth_hair = out['strands']
            save_hair(f'{outdir}/video/{fname}_frame_{idx:03d}.data', c2c(synth_hair.position[0]))
        for idx, (theta, beta) in enumerate(zip(theta_steps[::20], beta_steps[::20])):
            out = hair_model(roots=roots[None, ...], theta=theta[None, ...], beta=beta[None, ...])
            synth_image = out['image']
            video.append_data(np.concatenate([target_rgb, image_to_rgb(synth_image[0], rgb_size=256)], axis=1))
            synth_hair = out['strands']
            save_hair(f'{outdir}/video/{fname}_frame_{idx+len(warmup_steps)//20:03d}.data', c2c(synth_hair.position[0]))
        video.close()

    # Save final projected frame and W vector.
    theta = theta_steps[-1]
    beta = beta_steps[-1]
    out = hair_model(roots=roots[None, ...], theta=theta[None, ...], beta=beta[None, ...])
    synth_image = out['image']
    synth_hair = out['strands']
    synth_rgb = image_to_rgb(synth_image[0], rgb_size=256)
    PIL.Image.fromarray(target_rgb, 'RGB').save(f'{outdir}/{fname}_gt.png')
    PIL.Image.fromarray(synth_rgb, 'RGB').save(f'{outdir}/{fname}_proj.png')
    # np.savez(f'{outdir}/{fname}.npz', texture=c2c(synth_image[0]), roots=c2c(roots), theta=c2c(theta), beta=c2c(beta))
    np.savez(f'{outdir}/{fname}.npz', roots=c2c(roots), theta=c2c(theta), beta=c2c(beta))

    coords = hair_model.hair_roots.cartesian_to_spherical(roots)[..., :2]
    coords = hair_model.hair_roots.rescale(coords)
    target_hair = hair_model.G_res.sample(target_image[None, ...], coords[None, ...], mode='nearest')
    target_hair.position = target_hair.position + roots.unsqueeze(0).unsqueeze(2)
    save_hair(f'{outdir}/{fname}_gt.data', c2c(target_hair.position[0]))
    save_hair(f'{outdir}/{fname}_proj.data', c2c(synth_hair.position[0]))

    pos_diff = torch.norm(target_hair.position - synth_hair.position, dim=-1)
    cur_diff = (curvature(target_hair.position) - curvature(synth_hair.position)).abs()

    return c2c(pos_diff.mean()), c2c(cur_diff.mean())

# ----------------------------------------------------------------------------


@click.command()
@click.option('--model', 'model_path', help='Directory to load pre-trained model checkpoints', required=True)
@click.option('--target', help='Target hair neural texture to project to', required=True, metavar='FILE')
@click.option('--num-steps-warmup', help='Number of steps to warm-up the optimization', type=int, default=0, show_default=True)
@click.option('--num-steps', help='Number of optimization steps', type=int, default=500, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
def main(
    model_path: str,
    target: str,
    outdir: str,
    head_mesh: str,
    scalp_bounds: List[float],
    save_video: bool,
    seed: int,
    num_steps_warmup: int,
    num_steps: int
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
    device = torch.device('cuda')
    hair_model = Perm(model_path=model_path, head_mesh=head_mesh, scalp_bounds=scalp_bounds).eval().requires_grad_(False).to(device)

    if os.path.isdir(target):
        # target_files = sorted(glob.glob(os.path.join(target, '*.npz')))
        target_files = sorted(set(glob.glob(os.path.join(target, '*.npz'))) - set(glob.glob(os.path.join(target, '*_*.npz'))))
        metrics = dict(pos_diff=[], cur_diff=[])
        for f in target_files:
            pos_diff, cur_diff = run_projection(hair_model, f, outdir, num_steps_warmup, num_steps, save_video, device)
            metrics['pos_diff'].append(pos_diff)
            metrics['cur_diff'].append(cur_diff)

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(os.path.join(outdir, f'recon_metrics.csv'))
    else:
        pos_diff, cur_diff = run_projection(hair_model, target, outdir, num_steps_warmup, num_steps, save_video, device)
        print(f'pos diff: {pos_diff}')
        print(f'cur diff: {cur_diff}')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
