import glob
import os
import re
import subprocess
from typing import List, Union

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_cluster import nearest
from tqdm import tqdm

import dnnlib
from hair import HairRoots, Strands, load_hair, save_hair
from hair.loss import StrandGeometricLoss
from models import StrandCodec
from utils.blend_shape import sample, solve_blend_shapes
from utils.image import write_png, write_texture
from utils.misc import copy2cpu as c2c
from utils.misc import filename, load_tensor_dict
from utils.patch import split_patches
from utils.visualize import plot_explained_variance

DIST_THRESHOLD = 0.01


def hair_resampling(args, hair_roots, device):
    hair_files = sorted(glob.glob(os.path.join(args.indir, '*.data')))
    os.makedirs(os.path.join(args.outdir, 'data'), exist_ok=True)

    roots_resample, normal_resample = hair_roots.load_txt(args.roots)
    roots_resample = roots_resample.to(device)
    normal_resample = normal_resample.to(device)
    uv_resample = hair_roots.cartesian_to_spherical(roots_resample)[..., :2]

    DEGENERATED_LENGTH = 0.01

    for f in hair_files:
        fname = filename(f)
        strands = torch.tensor(load_hair(f), dtype=torch.float32, device=device)
        print(f'strands: {strands.shape}')

        roots = strands[:, 0].clone()
        strands -= roots.unsqueeze(1)

        # resample the number of strands for each hairstyle
        uv = hair_roots.cartesian_to_spherical(roots)[..., :2]
        index = nearest(uv_resample, uv)
        strands_resample = strands.index_select(dim=0, index=index)
        print(f'number of resampled strands: {strands_resample.shape[0]}')

        dist = torch.norm(uv_resample - uv.index_select(dim=0, index=index), dim=-1)
        if args.allow_degenerated:
            degenerated_index = torch.where(dist > DIST_THRESHOLD)[0]
            print(f'number of degenerated strands: {degenerated_index.shape[0]}')
            degenerated_dir = normal_resample.index_select(dim=0, index=degenerated_index)
            num_samples = strands.shape[1]
            length_index = torch.linspace(0, DEGENERATED_LENGTH, num_samples).to(device)
            strands_resample[degenerated_index] = length_index.reshape(1, num_samples, 1) * degenerated_dir.unsqueeze(1)
            strands_resample += hair_roots.spherical_to_cartesian(uv_resample).unsqueeze(1)
        else:
            strands_resample += hair_roots.spherical_to_cartesian(uv_resample).unsqueeze(1)
            valid_index = torch.where(dist <= DIST_THRESHOLD)[0]
            strands_resample = strands_resample.index_select(dim=0, index=valid_index)
            print(f'number of valid strands: {strands_resample.shape[0]}')

        save_hair(os.path.join(args.outdir, 'data', f'{fname}.data'), c2c(strands_resample))


def horizontal_flip(args):
    hair_files = sorted(glob.glob(os.path.join(args.indir, '*.data')))
    os.makedirs(args.outdir, exist_ok=True)

    for f in tqdm(hair_files):
        fname = filename(f)
        strands = load_hair(f)

        roots = strands[:, 0:1].copy()
        strands -= roots
        strands[..., 0] *= -1  # horizontally flip strand positions
        roots[..., 0] *= -1  # horizontally flip root positions
        strands += roots

        save_hair(os.path.join(args.outdir, f'{fname}_mirror.data'), strands)


def scalp_mask(args, hair_roots, device):
    roots, _ = hair_roots.load_txt(args.roots)
    roots = roots.to(device)
    roots = hair_roots.cartesian_to_spherical(roots)[..., :2]
    coords = hair_roots.rescale(roots)

    H = W = 256
    u, v = torch.meshgrid(torch.linspace(0, 1, steps=W, device=coords.device),
                          torch.linspace(0, 1, steps=H, device=coords.device), indexing='ij')
    uv = torch.dstack((u, v)).reshape(-1, 2)  # (W x H, 2)

    index = nearest(uv, coords)
    dist = torch.norm(coords.index_select(dim=0, index=index) - uv, dim=-1)
    mask = torch.where(dist > DIST_THRESHOLD, False, True)
    mask = mask.reshape(W, H, 1)  # (W, H, 1)
    mask_final = torch.logical_or(mask, mask.flip(0))
    mask_final = mask_final.transpose(0, 1)  # (H, W, 1)
    write_png('scalp_mask.png', mask_final)


def hair_blend_shapes(args):
    if 'strands' in args.bs_type:
        # NOTE: exclude mirrored and augmented data due to limited memory.
        # hair_files = sorted(set(glob.glob(os.path.join(args.indir, '*.data'))) - set(glob.glob(os.path.join(args.indir, '*_*.data'))))
        hair_files = sorted(glob.glob(os.path.join(args.indir, '*[!_mirror].data')))  # NOTE: exclude mirrored data due to limited memory.
        # hair_files = sorted(glob.glob(os.path.join(args.indir, '*.data')))
    else:
        hair_files = sorted(glob.glob(os.path.join(args.indir, '*.npz')))
    if args.range is not None:
        hair_files = hair_files[args.range[0]:args.range[1]]
    os.makedirs(args.outdir, exist_ok=True)

    data = []
    for f in tqdm(hair_files, desc='Loading data'):
        if 'strands' in args.bs_type:
            data.append(load_hair(f))
        else:
            data.append(np.load(f)['texture'])

    if 'strands' in args.bs_type:
        data = np.concatenate(data)
        data -= data[:, 0:1].copy()
        data = data[:, 1:]
        if args.fft:
            fourier = np.fft.rfft(data, n=data.shape[1], axis=-2, norm='ortho')
            data = np.concatenate([fourier.real, fourier.imag], axis=-1)
    else:
        data = np.stack(data)
        if args.bs_type == 'patch':
            patches = split_patches(data, patch_size=args.patch_size, overlap=False)
            data = patches.reshape(-1, patches.shape[2], args.patch_size, args.patch_size)

    print(f'data: {data.shape}')
    mean_shape = data.mean(axis=0, keepdims=True)
    blend_shapes, explained_variance_ratio = solve_blend_shapes(data - mean_shape, args.n_coeff, args.svd_solver)
    print(f'blend shapes: {blend_shapes.shape}')
    bs_type = args.bs_type.replace('_', '-')
    if args.fft:
        bs_type = 'fft-' + bs_type
    if args.bs_type == 'patch':
        bs_type = f'{args.patch_size}x{args.patch_size}-' + bs_type
    plot_explained_variance(os.path.join(args.outdir, f'{bs_type}-variance.png'), explained_variance_ratio)
    np.savez(os.path.join(args.outdir, f'{bs_type}-blend-shapes.npz'), mean_shape=mean_shape.astype(np.float32), blend_shapes=blend_shapes.astype(np.float32), variance_ratio=explained_variance_ratio)


def fit_neural_textures(coords, gt_strands, strand_codec, **opts):
    """ Fit explicit neural textures from projected hair strands.
    """
    # initialize neural textures with projected strand features
    u, v = torch.meshgrid(torch.linspace(0, 1, steps=opts['texture_size'], device=coords.device),
                          torch.linspace(0, 1, steps=opts['texture_size'], device=coords.device), indexing='ij')
    uv = torch.dstack((u, v)).reshape(-1, 2)  # (W x H, 2)

    index = nearest(uv, coords)
    dist = torch.norm(coords.index_select(dim=0, index=index) - uv, dim=-1)
    mask = torch.where(dist > DIST_THRESHOLD, False, True)

    strands = gt_strands.index_select(dim=0, index=index)
    strands = strands.filter(~mask)
    if strand_codec is None:
        init_texture = strands.to_tensor()
        init_texture = init_texture.flatten(-2)
    else:
        init_texture = strand_codec.encode(strands.position)

    init_texture = init_texture.reshape(1, opts['texture_size'], opts['texture_size'], -1)  # (1, W, H, C)
    init_texture = init_texture.permute(0, 3, 2, 1)  # (1, C, H, W)
    mask = mask.reshape(1, opts['texture_size'], opts['texture_size'], 1)  # (1, W, H, 1)
    mask = mask.permute(0, 3, 2, 1)  # (1, 1, H, W)

    # refine neural textures to resolve hash collisions and missing values
    texture = nn.Parameter(init_texture, requires_grad=True)
    optimizer = torch.optim.Adam([texture], lr=opts['lr'])
    criterion = StrandGeometricLoss()
    gt_strands.position = F.pad(gt_strands.position, (0, 0, 1, 0), mode='constant', value=0)

    for i in range(opts['iterations']):
        optimizer.zero_grad()
        coeff = sample(coords.unsqueeze(0), texture, opts['interp_mode'])[0]
        if strand_codec is None:
            position = coeff.reshape(coeff.shape[0], opts['samples_per_strand'], -1)
        else:
            position = strand_codec.decode(coeff)
        position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
        strands = Strands(position=position)
        strand_loss = criterion(strands, gt_strands)

        loss = 0
        log_text = f"STEP {i+1:04d}/{opts['iterations']}"
        for k, v in strand_loss.items():
            loss += v
            log_text += f" | strand_{k}_loss: {v.item():>.3E}"
        log_text += f" | loss: {loss.item():>.3E}"

        loss.backward()
        optimizer.step()

        if opts['verbose']:
            print(log_text)

    return texture[0], mask[0]


def hair_neural_textures(args, hair_roots, device):
    hair_files = sorted(glob.glob(os.path.join(args.indir, '*.data')))
    os.makedirs(args.outdir, exist_ok=True)

    if args.bsdir is not None:
        strand_codec = StrandCodec(args.bsdir, args.n_coeff, args.fft)
        strand_codec = strand_codec.to(device)
    else:
        strand_codec = None

    for f in tqdm(hair_files):
        torch.cuda.empty_cache()
        fname = filename(f)
        strands = torch.tensor(load_hair(f), dtype=torch.float32, device=device)
        roots = strands[:, 0].clone()
        roots_uv = hair_roots.cartesian_to_spherical(roots)
        coords = hair_roots.rescale(roots_uv[..., :2])

        position = strands - roots.unsqueeze(1)
        position = position[:, 1:]
        gt_strands = Strands(position=position)

        optim_args = dict(texture_size=args.size,
                          interp_mode=args.interp_mode,
                          samples_per_strand=99,
                          lr=0.001 if args.bsdir is not None else 0.0001,
                          iterations=500,
                          verbose=False
                          )
        texture, mask = fit_neural_textures(coords, gt_strands, strand_codec, **optim_args)

        np.savez(os.path.join(args.outdir, f'{fname}.npz'), texture=c2c(texture), mask=c2c(mask), roots=c2c(roots_uv))
        write_texture(os.path.join(args.outdir, f'{fname}.png'), texture.permute(1, 2, 0), alpha=mask.permute(1, 2, 0).float())


def guide_strand_textures(args, hair_roots, device):
    texture_files = sorted(glob.glob(os.path.join(args.indir, '*.npz')))
    # texture_files = [os.path.join(args.indir, 'strands00035.npz')]
    os.makedirs(args.outdir, exist_ok=True)

    if args.bsdir is not None:
        strand_codec = StrandCodec(args.bsdir, args.n_coeff, args.fft)
        strand_codec = strand_codec.to(device)
    else:
        strand_codec = None

    for f in tqdm(texture_files):
        torch.cuda.empty_cache()
        fname = filename(f)
        data = load_tensor_dict(f, device=device)
        texture = data['texture'][:args.n_coeff]
        mask = data['mask']
        u, v = torch.meshgrid(torch.linspace(0, 1, steps=texture.shape[-1], device=device),
                              torch.linspace(0, 1, steps=texture.shape[-2], device=device), indexing='ij')
        uv = torch.dstack((u, v)).permute(2, 1, 0)  # (2, H, W)

        texture_guide = F.interpolate(texture.unsqueeze(0), size=(args.size, args.size), mode='nearest')[0]
        mask_guide = F.interpolate(mask.unsqueeze(0), size=(args.size, args.size), mode='nearest')[0]
        uv_guide = F.interpolate(uv.unsqueeze(0), size=(args.size, args.size), mode='nearest')[0]

        uv_guide = uv_guide.permute(1, 2, 0).reshape(-1, 2)
        uv_guide = hair_roots.rescale(uv_guide, inverse=True)
        roots = hair_roots.spherical_to_cartesian(uv_guide)

        coeff = texture_guide.permute(1, 2, 0).reshape(-1, args.n_coeff)
        position = strand_codec.decode(coeff)
        position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
        position = position + roots.unsqueeze(1)
        save_hair(os.path.join(args.outdir, f'{fname}.data'), c2c(position))

        np.savez(os.path.join(args.outdir, f'{fname}.npz'), texture=c2c(texture_guide), mask=c2c(mask_guide), roots=c2c(uv_guide))
        write_texture(os.path.join(args.outdir, f'{fname}.png'), texture_guide.permute(1, 2, 0), alpha=mask_guide.permute(1, 2, 0).float())


def hair_canonicalization(args, hair_roots, device):
    texture_files = sorted(glob.glob(os.path.join(args.indir, 'high-res', '*.npz')))
    os.makedirs(args.outdir, exist_ok=True)

    strand_codec = StrandCodec(args.bsdir, fft=args.fft)
    strand_codec = strand_codec.to(device)

    for f in tqdm(texture_files):
        torch.cuda.empty_cache()
        fname = filename(f)
        data = np.load(f)
        roots = torch.tensor(data['roots'], dtype=torch.float32, device=device)
        coords = hair_roots.rescale(roots[..., :2]).unsqueeze(0)
        texture = torch.tensor(data['texture'][None, ...], dtype=torch.float32, device=device)
        guide_texture = np.load(os.path.join(args.indir, 'low-res', f'{fname}.npz'))['texture']
        guide_texture = torch.tensor(guide_texture[None, ...], dtype=torch.float32, device=device)
        guide_texture = F.interpolate(guide_texture, size=(texture.shape[0], texture.shape[1]), mode='nearest')

        coeff = sample(coords, texture, args.interp_mode)[0]
        guide_coeff = sample(coords, guide_texture, args.interp_mode)[0]
        guide_coeff = F.pad(guide_coeff, (0, coeff.shape[-1] - guide_coeff.shape[-1]), mode='constant', value=0)
        strands = strand_codec.decode(coeff)
        strands = F.pad(strands, (0, 0, 1, 0), mode='constant', value=0)
        strands = Strands(position=strands)
        guide_strands = strand_codec.decode(guide_coeff)
        guide_strands = F.pad(guide_strands, (0, 0, 1, 0), mode='constant', value=0)
        guide_strands = Strands(position=guide_strands)
        canonical_strands = strands.to_canonical(guide_strands)
        roots = hair_roots.spherical_to_cartesian(roots)
        # print(f'roots: {roots.shape}')
        position = canonical_strands.position + roots.unsqueeze(1)
        # print(f'position: {position.shape}')
        save_hair(os.path.join(args.outdir, f'{fname}.data'), c2c(position))


def fit_weight_images(k, raw_texture, gt_texture, strand_codec, **opts):
    """ Fit explicit neural textures from projected hair strands.
    """
    C, H, W = gt_texture.shape
    u, v = torch.meshgrid(torch.linspace(0, 1, steps=H, device=gt_texture.device),
                          torch.linspace(0, 1, steps=W, device=gt_texture.device), indexing='ij')
    uv = torch.dstack((u, v)).permute(2, 0, 1)  # (2, H, W)

    uv_guide = F.interpolate(uv.unsqueeze(0), size=(raw_texture.shape[1], raw_texture.shape[2]), mode='nearest')[0]  # (2, 32, 32)

    uv = uv.permute(1, 2, 0).reshape(-1, 2)  # (H x W, 2)
    uv_guide = uv_guide.permute(1, 2, 0).reshape(-1, 2)
    dist = torch.norm(uv.unsqueeze(1) - uv_guide.unsqueeze(0), dim=-1)  # (H x W, 32 x 32)
    knn_dist, knn_index = dist.topk(k, largest=False)

    guide_coeff = raw_texture.permute(1, 2, 0).reshape(-1, C)
    guide_coeff = guide_coeff.flatten(1).T.index_select(dim=-1, index=knn_index.flatten())
    guide_coeff = guide_coeff.reshape(1, -1, H, W, k)

    init_weight = torch.zeros(1, k, H, W, device=gt_texture.device)  # (1, K, H, W)
    init_weight[:, 0] = 1  # initialize to nearest upsampling
    weight_image = nn.Parameter(init_weight, requires_grad=True)
    optimizer = torch.optim.Adam([weight_image], lr=opts['lr'])

    for i in range(opts['iterations']):
        optimizer.zero_grad()
        texture = torch.einsum('nchwx,nxhw->nchw', guide_coeff, weight_image)
        loss = F.l1_loss(texture, gt_texture)
        log_text = f"STEP {i+1:04d}/{opts['iterations']}"
        log_text += f" | loss: {loss.item():>.3E}"

        loss.backward()
        optimizer.step()

        if opts['verbose']:
            print(log_text)

    return weight_image[0], texture[0]


def hair_weight_images(args, hair_roots, device):
    hair_files = sorted(glob.glob(os.path.join(args.indir, 'high-res', '*.npz')))[:1]
    os.makedirs(args.outdir, exist_ok=True)

    if args.bsdir is not None:
        strand_codec = StrandCodec(args.bsdir, args.n_coeff, args.fft)
        strand_codec = strand_codec.to(device)
    else:
        strand_codec = None

    for f in tqdm(hair_files):
        torch.cuda.empty_cache()
        fname = filename(f)
        data = load_tensor_dict(f, device=device)
        gt_texture = data['texture']
        roots = data['roots']
        raw_texture = np.load(os.path.join(args.indir, 'low-res', f'{fname}.npz'))['texture']
        raw_texture = torch.tensor(raw_texture, dtype=torch.float32, device=device)

        optim_args = dict(texture_size=args.size,
                          interp_mode=args.interp_mode,
                          samples_per_strand=99,
                          lr=0.007,
                          iterations=500,
                          verbose=True
                          )
        weight_image, texture = fit_weight_images(4, raw_texture, gt_texture, strand_codec, **optim_args)
        np.savez(os.path.join(args.outdir, f'{fname}.npz'), texture=c2c(weight_image))
        write_texture(os.path.join(args.outdir, f'{fname}.png'), texture.permute(1, 2, 0))


def hair_clustering(args):
    texture_files = sorted(glob.glob(os.path.join(args.indir, '*[!_mirror].npz')))
    os.makedirs(args.outdir, exist_ok=True)
    data = []
    for f in tqdm(texture_files):
        data.append(load_tensor_dict(f)['texture'])
    data = torch.stack(data)
    print(f'data: {data.shape}')
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init='auto').fit(c2c(data.reshape(len(texture_files), -1)))
    labels = kmeans.labels_

    data_files = map(lambda x: f'{filename(x)}.data', texture_files)
    data_files = np.array(list(data_files))
    print(f'data files: {data_files.shape}')
    for idx in range(args.n_clusters):
        file_idx = np.argwhere(labels == idx)
        filenames = np.take(data_files, file_idx)
        np.savetxt(os.path.join(args.outdir, f'{idx:02d}.txt'), filenames, fmt="%s")


def strand_coeff_normalization(args):
    texture_files = sorted(set(glob.glob(os.path.join(args.indir, '*.npz'))) - set(glob.glob(os.path.join(args.indir, '*_*.npz'))))

    data = []
    for f in tqdm(texture_files):
        data.append(load_tensor_dict(f)['texture'])
    data = torch.stack(data)
    print(f'data: {data.shape}')
    mean = data.mean(dim=(0, 2, 3))
    std = data.std(dim=(0, 2, 3))
    std[std == 0] = 1
    print(f'mean: {mean.shape}')
    print(f'std:  {std.shape}')

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, 'coeff-stats.npz')
    np.savez(outfile, mean=c2c(mean), std=c2c(std))
    print(f'Save mean and std file to {outfile}')


def strand_filtering(args):
    hair_files = sorted(glob.glob(os.path.join(args.indir, '*.data')))
    os.makedirs(args.outdir, exist_ok=True)

    for f in hair_files:
        fname = filename(f)
        strands = load_hair(f)
        print(f'original strands: {strands.shape}')
        segments = np.linalg.norm(strands[:, 1:] - strands[:, :-1], axis=-1)
        lengths = segments.sum(axis=-1)
        index = np.where(lengths > args.length, True, False)
        filtered_strands = strands[index, ...]
        print(f'filtered strands: {filtered_strands.shape}')
        save_hair(os.path.join(args.outdir, f'{fname}.data'), filtered_strands)


def convert(args):
    hair_files = sorted(glob.glob(os.path.join(args.indir, '*.data')))
    os.makedirs(args.outdir, exist_ok=True)

    for f in tqdm(hair_files):
        fname = filename(f)
        strands = load_hair(f)
        save_hair(os.path.join(args.outdir, f'{fname}.obj'), strands)

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
# Required.
@click.option('--process_fn', '-p', help='Data processing function.', metavar='STR', type=click.Choice(['resample', 'scalp', 'flip', 'blend_shapes', 'texture', 'guide_strands', 'canonical', 'weight_image', 'normalize', 'cluster', 'filter', 'convert']), required=True)
@click.option('--indir', '-i', help='Where to load the data.', metavar='DIR', required=True)
@click.option('--range', type=parse_range, help='Range of input data to load (e.g., \'0,1,4-6\')', required=False)
@click.option('--outdir', '-o', help='Where to save the results.', metavar='DIR', required=True)
# Hair-related parameters.
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str, default='./data/head.obj', show_default=True)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
# Hair blend shapes.
@click.option('--bs_type', help='Type of blend shapes to solve.', metavar='STR', type=click.Choice(['strands', 'raw_tex', 'patch']))
@click.option('--n_coeff', help='Number of PCA coefficients.', metavar='INT', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--svd_solver', help='SVD solver for PCA.', metavar='STR', type=click.Choice(['full', 'auto']), default='full', show_default=True)
@click.option('--fft', help='Whether to solve blend shapes in the frequency domain or the spatial domain.', metavar='BOOL', type=bool, default=False, show_default=True)
# Hair neural textures.
@click.option('--texture_type', help='Type of neural textures to fit.', metavar='STR', type=click.Choice(['guide', 'local', 'strands', 'pca']))
@click.option('--interp_mode', help='Texture interpolation mode for sampling.', metavar='STR', type=click.Choice(['nearest', 'bilinear']))
@click.option('--bsdir', help='Where to load the pre-computed blend shapes.', metavar='DIR')
@click.option('--size', help='Output size of hair neural textures.', metavar='INT', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--patch_size', help='Patch size to crop from neural textures.', metavar='INT', type=click.IntRange(min=1), default=8, show_default=True)
# Hair resampling.
@click.option('--roots', metavar='DIR', help='Path to the hair root file for resampling.', required=False)
@click.option('--allow_degenerated', help='Whether allow degenerated strands in the resampled hair (whose length would be 0.01).', metavar='BOOL', type=bool, default=False, show_default=True)
# Hair clustering.
@click.option('--n_clusters', help='Number of clusters to form.', metavar='INT', type=click.IntRange(min=1), default=10, show_default=True)
# Strand filtering.
@click.option('--length', help='Threshold of length to filter.', metavar='FLOAT', type=click.FloatRange(min=0.0), default=0.1, show_default=True)
def main(**kwargs):
    # Initialize.
    args = dnnlib.EasyDict(kwargs)  # Command line arguments.
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hair_roots = HairRoots(head_mesh=args.head_mesh, scalp_bounds=args.scalp_bounds)

    if args.process_fn == 'flip':
        horizontal_flip(args)
    elif args.process_fn == 'blend_shapes':
        hair_blend_shapes(args)
    elif args.process_fn == 'texture':
        hair_neural_textures(args, hair_roots, device)
    elif args.process_fn == 'guide_strands':
        guide_strand_textures(args, hair_roots, device)
    elif args.process_fn == 'canonical':
        hair_canonicalization(args, hair_roots, device)
    elif args.process_fn == 'weight_image':
        hair_weight_images(args, hair_roots, device)
    elif args.process_fn == 'normalize':
        strand_coeff_normalization(args)
    elif args.process_fn == 'cluster':
        hair_clustering(args)
    elif args.process_fn == 'filter':
        strand_filtering(args)
    elif args.process_fn == 'convert':
        convert(args)
    elif args.process_fn == 'resample':
        hair_resampling(args, hair_roots, device)
    elif args.process_fn == 'scalp':
        scalp_mask(args, hair_roots, device)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
