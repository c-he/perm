import copy
import os

import torch

import dnnlib
from hair import HairRoots, save_hair
from utils.image import write_texture
from utils.misc import EPSILON
from utils.misc import copy2cpu as c2c

from . import metric_utils

# ----------------------------------------------------------------------------


def curvature(position):
    # we first compute the circumradius r for every 3 adjacent points on the strand, and curvature is defined as 1/r.
    # https://en.wikipedia.org/wiki/Circumscribed_circle
    a = position[..., :-2, :] - position[..., 2:, :]  # (..., num_samples - 2, 3)
    b = position[..., 1:-1, :] - position[..., 2:, :]  # (..., num_samples - 2, 3)
    c = a - b
    curvature = 2.0 * torch.norm(torch.cross(a, b, dim=-1), dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1) * torch.norm(c, dim=-1) + EPSILON)  # (batch_size, num_strands, num_samples - 2)

    return curvature

# ----------------------------------------------------------------------------


def compute_stats_for_hair_reconstruction(opts, rel_lo=0, rel_hi=1, batch_size=1, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Setup generator.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    hair_roots = HairRoots(**opts.hair_roots_kwargs)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    pos_stats = metric_utils.FeatureStats(max_items=num_items, **stats_kwargs)
    cur_stats = metric_utils.FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    os.makedirs(opts.output_dir, exist_ok=True)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for hair, gt_texture, idx in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        idx = idx.to(opts.device)
        z = G.lookup(idx.view(-1).long())
        # texture = G(z, **opts.G_kwargs)
        gen_textures = G(z, **opts.G_kwargs)
        texture = gen_textures['texture']
        raw_texture = gen_textures['raw_texture']
        coordinates = hair_roots.scale(hair['roots'][..., :2]).to(opts.device)
        strands = G.sample(texture, coordinates)
        gt_texture = gt_texture.to(opts.device)
        gt_strands = G.sample(gt_texture, coordinates)

        # if opts.hair_type == 'local':
        #     gt_position = hair['local_strands']
        # elif opts.hair_type == 'guide':
        #     gt_position = hair['guide_strands']
        #     gt_position = gt_position.index_select(dim=0, index=hair['labels'].int())
        # else:
        #     gt_position = hair['strands']
        # gt_position = gt_position - gt_position[:, :, 0:1].clone()
        # gt_position = gt_position.to(opts.device)
        pos_diff = torch.norm(strands.position - gt_strands.position, dim=-1)
        # print(f'pos_diff: {pos_diff.shape}')
        pos_diff = pos_diff.flatten(1).mean(dim=1, keepdim=True)
        pos_stats.append_torch(pos_diff, num_gpus=opts.num_gpus, rank=opts.rank)
        cur_diff = (curvature(strands.position) - curvature(gt_strands.position)).abs()
        # print(f'cur_diff: {cur_diff.shape}')
        cur_diff = cur_diff.flatten(1).mean(dim=1, keepdim=True)
        cur_stats.append_torch(cur_diff, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(pos_stats.num_items)
        if opts.output_texture:
            write_texture(os.path.join(opts.output_dir, f'texture{idx.item():04d}.png'), texture[0].permute(1, 2, 0), normalize=True)
            write_texture(os.path.join(opts.output_dir, f'texture{idx.item():04d}_raw.png'), raw_texture[0].permute(1, 2, 0), normalize=True)
            write_texture(os.path.join(opts.output_dir, f'texture{idx.item():04d}_gt.png'), gt_texture[0].permute(1, 2, 0), normalize=True)
        if opts.output_geometry:
            roots = hair_roots.spherical_to_cartesian(hair['roots']).to(opts.device)
            strands.position = strands.position + roots.unsqueeze(2)
            gt_strands.position = gt_strands.position + roots.unsqueeze(2)
            save_hair(os.path.join(opts.output_dir, f'strands{idx.item():04d}.ply'), c2c(strands.position[0]))
            save_hair(os.path.join(opts.output_dir, f'strands{idx.item():04d}_gt.ply'), c2c(gt_strands.position[0]))

    return float(pos_stats.get_all().mean()), float(cur_stats.get_all().mean())

# ----------------------------------------------------------------------------
