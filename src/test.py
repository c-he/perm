import numpy as np
import torch
import torch.nn.functional as F

from hair import HairRoots, load_hair, save_hair
from models import StrandCodec
from utils.blend_shape import sample
from utils.image import write_texture, write_png
from utils.misc import load_tensor_dict

# data = np.load('/disk3/proj_hair/hair_sim_data/Wind/strands00001/npz/strands00001_0.npz')['strands']
# data = torch.tensor(data, dtype=torch.float32)
# print(data.shape)
# write_texture('test.png', data.permute(1, 2, 0), normalize=True)

# fname = 'strands00001'
# device = torch.device('cuda:0')

# hair_roots = HairRoots('data/head.obj', scalp_bounds=[0.1870, 0.8018, 0.4011, 0.8047])
# strand_codec = StrandCodec('data/blend-shapes/fft-strands-blend-shapes.npz', num_coeff=64, fft=True).to(device)

# strands = torch.tensor(load_hair(f'data/usc-hair/{fname}.data'), dtype=torch.float32)
# print(f'strands: {strands.shape}')
# roots = strands[:, 0].clone()
# strands = strands - roots.unsqueeze(1)
# strands_recon = strand_codec(strands[:, 1:])
# strands_recon = F.pad(strands_recon, (0, 0, 1, 0), mode='constant', value=0)
# strands_recon = strands_recon + roots.unsqueeze(1)
# save_hair('strands00001.abc', strands.cpu().numpy())

# data = np.load(f'data/neural-textures/high-res/{fname}.npz')
# gt_texture = torch.tensor(data['texture'], dtype=torch.float32, device=device)
# print(f'gt_texture: {gt_texture.shape}')
# roots = torch.tensor(data['roots'], dtype=torch.float32, device=device)
# gt_coords = hair_roots.rescale(roots[..., :2])
# print(f'gt_coords: {gt_coords.shape}')

# gt_coeff = sample(gt_coords.unsqueeze(0), gt_texture.unsqueeze(0), mode='nearest')[0]
# gt_position = strand_codec.decode(gt_coeff)
# gt_position = F.pad(gt_position, (0, 0, 1, 0), mode='constant', value=0)
# gt_position = gt_position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
# print(f'gt_position: {gt_position.shape}')
# save_hair(f'{fname}-sample.abc', gt_position.cpu().numpy())

# roots_20k, _ = hair_roots.load_txt('data/roots/rootPositions_20k.txt')
# print(f'roots_20k: {roots_20k.shape}')
# roots_20k = hair_roots.cartesian_to_spherical(roots_20k)
# roots_20k = roots_20k.to(device)
# coords_20k = hair_roots.rescale(roots_20k[..., :2])
# print(f'coords_20k: {coords_20k.shape}')

# coeff = sample(coords_20k.unsqueeze(0), gt_texture.unsqueeze(0), mode='nearest')[0]
# position = strand_codec.decode(coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + hair_roots.spherical_to_cartesian(roots_20k).unsqueeze(1)
# print(f'position: {position.shape}')
# save_hair('strands00001-20k.abc', position.cpu().numpy())

# from utils.blend_shape import blend, project
# from utils.patch import split_patches, assemble_patches

# bs_data = np.load('data/blend-shapes/low-res-tex-blend-shapes.npz')
# mean_shape = torch.tensor(bs_data['mean_shape'], dtype=torch.float32)
# blend_shapes = torch.tensor(bs_data['blend_shapes'], dtype=torch.float32)
# print(f'mean_shape: {mean_shape.shape}')
# print(f'blend_shapes: {blend_shapes.shape}')
# data = load_tensor_dict('data/neural-textures/low-res/strands00001.npz')
# image = data['texture']
# print(f'image: {image.shape}')
# coeff = project(image.unsqueeze(0) - mean_shape, blend_shapes)
# print(f'coeff: {coeff.shape}')
# image_recon = mean_shape + blend(coeff, blend_shapes)
# print(f'image_recon: {image_recon.shape}')
# write_texture('test.png', image_recon[0].permute(1, 2, 0))
# roots = data['roots']
# roots = hair_roots.spherical_to_cartesian(roots)
# print(f'roots: {roots.shape}')
# coeff = image_recon[0].permute(1, 2, 0).reshape(-1, 64)
# position = strand_codec.decode(coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + roots.unsqueeze(1)
# print(f'position: {position.shape}')
# save_hair(f'strands00001-guide-recon.ply', position.cpu().numpy())

# bs_data = np.load('data/blend-shapes/4x4-patch-blend-shapes.npz')
# mean_shape = torch.tensor(bs_data['mean_shape'], dtype=torch.float32)
# blend_shapes = torch.tensor(bs_data['blend_shapes'], dtype=torch.float32)
# print(f'mean_shape: {mean_shape.shape}')
# print(f'blend_shapes: {blend_shapes.shape}')
# data = load_tensor_dict('data/neural-textures/low-res/strands00001.npz')
# image = data['texture']
# patches = split_patches(image.unsqueeze(0), patch_size=4, overlap=False)
# patches = patches.reshape(-1, patches.shape[2], 4, 4)
# print(f'patches: {patches.shape}')
# coeff = project(patches - mean_shape, blend_shapes)
# print(f'coeff: {coeff.shape}')
# patches_recon = mean_shape + blend(coeff, blend_shapes)
# print(f'patches_recon: {patches_recon.shape}')
# image_recon = assemble_patches(patches_recon.unsqueeze(1), overlap=False)
# write_texture('test.png', image_recon[0].permute(1, 2, 0))
# roots = data['roots']
# roots = hair_roots.spherical_to_cartesian(roots)
# print(f'roots: {roots.shape}')
# coeff = image_recon[0].permute(1, 2, 0).reshape(-1, 64)
# position = strand_codec.decode(coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + roots.unsqueeze(1)
# print(f'position: {position.shape}')
# save_hair(f'strands00001-guide-recon.ply', position.cpu().numpy())

# beta = torch.randn(10, 512) * 10
# image_raw = mean_shape + blend(beta, blend_shapes)
# print(f'image_raw: {image_raw.shape}')
# for i in range(10):
#     coeff = image_raw[i].permute(1, 2, 0).reshape(-1, 64)
#     position = strand_codec.decode(coeff)
#     position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
#     position = position + roots.unsqueeze(1)
#     print(f'position: {position.shape}')
#     save_hair(f'guide-strands-sample-{i}.ply', position.cpu().numpy())

# data = np.load('data/neural-textures/low-res/strands00001.npz')
# texture = torch.tensor(data['texture'], dtype=torch.float32, device=device)
# print(f'texture: {texture.shape}')

# u, v = torch.meshgrid(torch.linspace(0, 1, steps=256, device=device),
#                       torch.linspace(0, 1, steps=256, device=device), indexing='ij')
# uv = torch.dstack((u, v)).permute(2, 1, 0)  # (2, 256, 256)
# uv_guide = F.interpolate(uv.unsqueeze(0), size=(32, 32), mode='nearest')[0]  # (2, 32, 32)
# k = 4

# uv = uv.permute(1, 2, 0).reshape(-1, 2)  # (H x W, 2)
# uv_guide = uv_guide.permute(1, 2, 0).reshape(-1, 2)
# dist = torch.norm(uv.unsqueeze(1) - uv_guide.unsqueeze(0), dim=-1)  # (256 x 256, 32 x 32)
# print(f'dist: {dist.shape}')
# knn_dist, knn_index = dist.topk(k, largest=False)
# print(knn_index.shape)
# print(knn_index)
# print(knn_dist.shape)

# guide_coeff = texture.reshape(64, -1)
# guide_coeff = guide_coeff.index_select(dim=-1, index=knn_index.flatten())
# guide_coeff = guide_coeff.reshape(1, -1, 256, 256, k)
# print(f'guide_coeff: {guide_coeff.shape}')

# weight_image = np.load('data/test/strands00001.npz')['texture']
# weight_image = torch.tensor(weight_image[None, ...], dtype=torch.float32, device=device)
# print(f'weight_image: {weight_image.shape}')

# texture = torch.einsum('nchwx,nxhw->nchw', guide_coeff, weight_image)
# coeff = sample(gt_coords.unsqueeze(0), texture, mode='nearest')[0]
# # fourier = fourier.reshape(gt_coords.shape[0], -1, 6)
# # print(f'fourier: {fourier.shape}')
# # position = torch.fft.irfft(torch.complex(fourier[..., :3], fourier[..., 3:]), n=strand_codec.SAMPLES_PER_STRAND - 1, dim=-2, norm='ortho')
# position = strand_codec.decode(coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
# print(f'position: {position.shape}')
# gt_coeff = sample(gt_coords.unsqueeze(0), gt_texture.unsqueeze(0), mode='nearest')[0]
# gt_position = strand_codec.decode(gt_coeff)
# gt_position = F.pad(gt_position, (0, 0, 1, 0), mode='constant', value=0)
# gt_position = gt_position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
# print(f'pos diff: {F.l1_loss(position, gt_position)}')
# save_hair('strands00001-10k.ply', position.cpu().numpy())
# write_texture('test.png', texture[0].permute(1, 2, 0))

# data = np.load('data/neural-textures/low-res/strands00035.npz')
# low_res = torch.tensor(data['texture'], dtype=torch.float32, device=device)
# print(f'low_res: {low_res.shape}')
# data = np.load('data/neural-textures/high-res/strands00035.npz')
# high_res = torch.tensor(data['texture'], dtype=torch.float32, device=device)
# print(f'high_res: {high_res.shape}')
# upsampled = F.interpolate(low_res.unsqueeze(0), (256, 256), mode='nearest')
# upsampled = F.pad(upsampled, (0, 0, 0, 0, 0, 54), mode='constant', value=0)[0]
# print(f'upsampled: {upsampled.shape}')
# residual = high_res - upsampled
# write_texture('residual.png', residual.permute(1, 2, 0))

# data = np.load('data/neural-textures/low-res/strands00001.npz')
# low_res = torch.tensor(data['texture'], dtype=torch.float32, device=device)
# upsampled = F.interpolate(low_res.unsqueeze(0), (256, 256), mode='nearest')
# upsampled = F.pad(upsampled, (0, 0, 0, 0, 0, 54), mode='constant', value=0)[0]
# texture = residual + upsampled
# texture[:10] = upsampled[:10]
# write_texture('test.png', texture.permute(1, 2, 0))

# data_low_rank = np.load('data/neural-textures/high-res/strands00002.npz')
# data_high_rank = np.load('data/neural-textures/high-res/strands00035.npz')

# low_rank_coeff = torch.tensor(data_low_rank['texture'][None, ...], dtype=torch.float32, device=device)
# low_rank_coeff = low_rank_coeff[:, :10]
# high_rank_coeff = torch.tensor(data_high_rank['texture'][None, ...], dtype=torch.float32, device=device)
# high_rank_coeff = high_rank_coeff[:, 10:]
# coeff_image = torch.cat([low_rank_coeff, high_rank_coeff], dim=1)

# roots = torch.tensor(data_low_rank['roots'], dtype=torch.float32, device=device)
# coords = hair_roots.rescale(roots[..., :2])

# coeff = sample(coords.unsqueeze(0), coeff_image, mode='nearest')[0]
# position = strand_codec.decode(coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
# print(f'position: {position.shape}')
# save_hair('strands00002_strands00035.abc', position.cpu().numpy())

# data_low_rank = np.load('data/neural-textures/low-res/strands00001.npz')
# data_high_rank = np.load('data/neural-textures/high-res/strands00001.npz')

# low_rank_coeff = torch.tensor(data_low_rank['texture'][None, ...], dtype=torch.float32, device=device)
# low_rank_coeff = F.interpolate(low_rank_coeff, (256, 256), mode='bilinear', align_corners=False)
# high_rank_coeff = torch.tensor(data_high_rank['texture'][None, ...], dtype=torch.float32, device=device)
# high_rank_coeff = high_rank_coeff[:, 10:]

# low_rank_coeff = F.pad(low_rank_coeff, (0, 0, 0, 0, 0, 54), mode='constant', value=0)
# high_rank_coeff = F.pad(high_rank_coeff, (0, 0, 0, 0, 10, 0), mode='constant', value=0)
# print(f'low_rank_coeff: {low_rank_coeff}')
# print(f'high_rank_coeff: {high_rank_coeff}')
# coeff_image = low_rank_coeff + high_rank_coeff

# roots = torch.tensor(data_high_rank['roots'], dtype=torch.float32, device=device)
# coords = hair_roots.rescale(roots[..., :2])

# coeff = sample(coords.unsqueeze(0), coeff_image, mode='nearest')[0]
# position = strand_codec.decode(coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
# print(f'position: {position.shape}')
# save_hair('strands00001-upsample.ply', position.cpu().numpy())

# data = np.load('data/neural-textures/high-res/strands00002.npz')
# texture = torch.tensor(data['texture'][None, ...], dtype=torch.float32, device=device)
# roots = torch.tensor(data['roots'], dtype=torch.float32, device=device)
# coords = hair_roots.rescale(roots[..., :2])

# coeff = sample(coords.unsqueeze(0), texture, mode='nearest')[0]
# position = strand_codec.decode(coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
# print(f'position: {position.shape}')
# save_hair('strands00002-sample.abc', position.cpu().numpy())
# low_rank_coeff = coeff[:, :10]
# low_rank_coeff = F.pad(low_rank_coeff, (0, 54), mode='constant', value=0)
# print(low_rank_coeff.shape)
# position = strand_codec.decode(low_rank_coeff)
# position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
# position = position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
# print(f'position: {position.shape}')
# save_hair('strands00035-smooth.abc', position.cpu().numpy())

# data = np.load('data/neural-textures/high-res/strands00129.npz')
# texture = torch.tensor(data['texture'], dtype=torch.float32)
# mask = torch.tensor(data['mask'], dtype=torch.float32)
# write_texture('rendering/overview/strands00129.png', texture=texture.permute(1, 2, 0))
# write_texture('rendering/overview/strands00129_residual.png', texture=texture[10:].permute(1, 2, 0))
# write_png('rendering/overview/strands00129_mask.png', mask.permute(1, 2, 0))
# data = np.load('data/neural-textures/low-res/strands00129.npz')
# texture = torch.tensor(data['texture'], dtype=torch.float32)
# mask = torch.tensor(data['mask'], dtype=torch.float32)
# texture = F.interpolate(texture.unsqueeze(0), (256, 256), mode='nearest')[0]
# mask = F.interpolate(mask.unsqueeze(0), (256, 256), mode='nearest')[0]
# write_texture('rendering/overview/strands00129_raw.png', texture=texture.permute(1, 2, 0))
# write_png('rendering/overview/strands00129_raw_mask.png', mask.permute(1, 2, 0))

# strands = torch.zeros([100, 100, 3], dtype=torch.float32, device=device)
# t = torch.linspace(0, -10, 100)
# amp = 0.3
# freq = 2.0
# strands[..., 0] = (t * freq).sin() * amp
# strands[..., 1] = (t * freq).cos() * amp
# strands[..., 2] = t

# R = 0.5
# r = R * torch.rand(100).sqrt()
# theta = torch.rand(100) * 2 * np.pi

# roots = torch.zeros(100, 1, 3, dtype=torch.float32, device=device)
# roots[:, 0, 0] = r * theta.cos()
# roots[:, 0, 1] = r * theta.sin()
# strands += roots
# print(strands)
# save_hair('test.abc', data=strands.cpu().numpy())

from utils.metric import curvature
from hair import load_hair

import glob
import pandas as pd
from tqdm import tqdm

usc_hair = sorted(glob.glob('data/usc-hair*/*.data'))
cur = []
for f in tqdm(usc_hair):
    # print(f)
    strands = load_hair(f)
    c = curvature(strands)
    cur.append(c.mean())
df = pd.DataFrame(cur)
df.to_csv('cur_stats.csv', index=False)
