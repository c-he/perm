# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Code adapted from
"Alias-Free Generative Adversarial Networks"."""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from torch_utils import training_stats
from torch_utils import custom_ops

# ----------------------------------------------------------------------------


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

# ----------------------------------------------------------------------------


def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size}')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

# ----------------------------------------------------------------------------


def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.NeuralTextureDataset', path=data)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)  # Subclass of training.dataset.Dataset.
        dataset_kwargs.img_resolution = dataset_obj.img_resolution           # Be explicit about resolution.
        dataset_kwargs.raw_resolution = dataset_obj.raw_resolution           # Be explicit about resolution.
        dataset_kwargs.max_size = len(dataset_obj)                           # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

# ----------------------------------------------------------------------------


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

# ----------------------------------------------------------------------------


@click.command()
# Required.
@click.option('--outdir', help='Where to save the results', metavar='DIR', required=True)
@click.option('--cfg', help='Base configuration', type=click.Choice(['raw-tex', 'res-tex', 'super-res']), required=True)
@click.option('--data', help='Path to hair neural textures', metavar='DIR', type=str, required=True)
@click.option('--gpus', help='Number of GPUs to use', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--batch', help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
# Optional features.
@click.option('--gamma', help='R1 regularization weight', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--gamma-mask', help='R1 regularization weight for binary mask', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--aug', help='Augmentation mode', type=click.Choice(['noaug', 'ada', 'fixed']), default='noaug', show_default=True)
@click.option('--resume', help='Resume from given network pickle', metavar='[PATH|URL]', type=str)
@click.option('--freezed', help='Freeze first layers of D', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
# Misc hyperparameters.
@click.option('--p', help='Probability for --aug=fixed', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target', help='Target value for --aug=ada', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu', help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1))
@click.option('--cbase', help='Capacity multiplier', metavar='INT', type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax', help='Max. feature maps', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr', help='G learning rate  [default: varies]', metavar='FLOAT', type=click.FloatRange(min=0))
@click.option('--dlr', help='D learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth', help='Mapping network depth  [default: varies]', metavar='INT', type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--mbstd-group', help='Minibatch std group size', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--blur_fade_kimg', help='Blur over how many', metavar='INT', type=click.IntRange(min=0), required=False, default=200)
@click.option('--resume_blur', help='Enable to blur even on resume', metavar='BOOL', type=bool, required=False, default=False)
# Neural texture settings.
# @click.option('--interp_mode', help='Interpolation mode to sample neural textures', metavar='STR', type=click.Choice(['nearest', 'bilinear']), default='nearest', show_default=True)
@click.option('--strand_codec', help='Path to pre-computed strand PCA blend shapes', metavar='DIR', type=str)
@click.option('--n_coeff', help='Number of PCA coefficients.', metavar='INT', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--fft', help='Whether the blend shapes are solved in the frequency domain or the spatial domain.', metavar='BOOL', type=bool, default=False, show_default=True)
# Super resolution settings.
@click.option('--encode_space', help='Output latent space for encoder', metavar='STR', type=click.Choice(['w', 'w+']), default='w', show_default=True)
@click.option('--sr_mode', help='Output mode for super resolution', metavar='STR', type=click.Choice(['coeff', 'weight', 'hybrid']), default='hybrid', show_default=True)
@click.option('--lambda_tex', help='Strength for texture reconstruction', metavar='FLOAT', type=click.FloatRange(min=0), required=False, default=1, show_default=True)
@click.option('--lambda_geo', help='Strength for strand geometric reconstruction', metavar='FLOAT', type=click.FloatRange(min=0), required=False, default=0, show_default=True)
@click.option('--lambda_reg', help='Strength for coeff regularization', metavar='FLOAT', type=click.FloatRange(min=0), required=False, default=0, show_default=True)
@click.option('--lambda_kl', help='Strength for KL divergence', metavar='FLOAT', type=click.FloatRange(min=0), required=False, default=0, show_default=True)
# Hair-related parameters.
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str, default='./data/head.obj', show_default=True)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
@click.option('--roots', metavar='DIR', help='Path to the hair root file.', type=str, default=None)
# Misc settings.
@click.option('--desc', help='String to include in result dir name', metavar='STR', type=str)
@click.option('--metrics', help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='none', show_default=True)
@click.option('--kimg', help='Total training duration', metavar='KIMG', type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick', help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap', help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed', help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--nobench', help='Disable cuDNN benchmarking', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--workers', help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n', '--dry-run', help='Print training options and exit', is_flag=True)
@click.option('--g_num_fp16_res', help='Number of fp16 layers in generator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
@click.option('--d_num_fp16_res', help='Number of fp16 layers in discriminator', metavar='INT', type=click.IntRange(min=0), default=0, required=False, show_default=True)
def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = dnnlib.EasyDict(kwargs)         # Command line arguments.
    c = dnnlib.EasyDict(exp_cfg=opts.cfg)  # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), strand_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='models.discriminator.DualDiscriminator', c_dim=0, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0, 0.99], eps=1e-8)
    c.loss_kwargs = dnnlib.EasyDict(class_name=None)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)
    c.hair_roots_kwargs = dnnlib.EasyDict(class_name='hair.hair_roots.HairRoots', head_mesh=opts.head_mesh, scalp_bounds=opts.scalp_bounds)
    c.roots = opts.roots

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)

    # Hyperparameters & settings.
    opts.gpus = min(opts.gpus, torch.cuda.device_count())
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.G_opt_kwargs.lr = 0.002 if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'raw-tex':
        c.G_kwargs.class_name = 'models.neural_texture.RawNeuralTexture'
        c.D_kwargs.class_name = 'models.discriminator.DualDiscriminator'
        c.D_kwargs.epilogue_kwargs.class_name = 'models.networks_stylegan2.DiscriminatorEpilogue'
        c.D_kwargs.aux_img_type = 'mask'
        c.loss_kwargs.class_name = 'training.loss.StyleGAN2Loss'
        c.loss_kwargs.r1_gamma_mask = opts.gamma_mask
    elif opts.cfg == 'res-tex':
        c.G_kwargs.class_name = 'models.neural_texture.ResNeuralTexture'
        c.G_kwargs.img_scale = True
        c.G_kwargs.mapping_kwargs.space = opts.encode_space
        c.loss_kwargs.class_name = 'training.loss.VAELoss'
        c.loss_kwargs.lambda_tex = opts.lambda_tex
        c.loss_kwargs.lambda_geo = opts.lambda_geo
        c.loss_kwargs.lambda_kl = opts.lambda_kl
    elif opts.cfg == 'super-res':
        c.G_kwargs.class_name = 'models.neural_texture.NeuralTextureSuperRes'
        c.G_kwargs.sr_mode = opts.sr_mode
        c.loss_kwargs.class_name = 'training.loss.UNetSuperResLoss'
        c.loss_kwargs.lambda_tex = opts.lambda_tex
        c.loss_kwargs.lambda_geo = opts.lambda_geo
        c.loss_kwargs.lambda_reg = opts.lambda_reg
    else:
        assert False, "Need to specify config"
    c.G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.

    # Neural texture configuration.
    c.G_kwargs.strand_kwargs.model_path = opts.strand_codec  # Path to the pre-computed strand PCA blend shapes.
    c.G_kwargs.strand_kwargs.num_coeff = opts.n_coeff        # Number of PCA coefficients.
    c.G_kwargs.strand_kwargs.fft = opts.fft                  # Whether the blend shapes are solved in the frequency domain or the spatial domain.

    c.loss_kwargs.blur_init_sigma = 10  # Blur the images seen by the discriminator.
    c.loss_kwargs.blur_fade_kimg = c.batch_size * opts.blur_fade_kimg / 32  # Fade out the blur during the first N kimg.

    # Augmentation.
    if opts.aug != 'noaug':
        # c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1,
        #                                    rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', imgfilter=1, imgrescale=1, noise=1)
        if opts.aug == 'ada':
            c.ada_target = opts.target
        if opts.aug == 'fixed':
            c.augment_p = opts.p

    # Resume.
    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100     # Make ADA react faster at the beginning.
        c.ema_rampup = None  # Disable EMA rampup.
        if not opts.resume_blur:
            c.loss_kwargs.blur_init_sigma = 0  # Disable blur rampup.

    # Performance-related toggles.
    # if opts.fp32:
    #     c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
    #     c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    c.G_kwargs.num_fp16_res = opts.g_num_fp16_res
    c.G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None
    c.D_kwargs.num_fp16_res = opts.d_num_fp16_res
    c.D_kwargs.conv_clamp = 256 if opts.d_num_fp16_res > 0 else None

    if opts.nobench:
        c.cudnn_benchmark = False

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
