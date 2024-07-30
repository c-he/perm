import glob
import os
import re
from typing import List, Union

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hair import HairRoots, save_hair
from models import StrandCodec
from utils.blend_shape import blend, project
from utils.image import write_texture
from utils.metric import curvature
from utils.misc import copy2cpu as c2c
from utils.misc import load_tensor_dict

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
@click.option('--eval_mode', help='Evaluation mode for PCA blend shapes', metavar='STR', type=click.Choice(['sample', 'recon']), required=True)
@click.option('--indir', '-i', help='Where to load the data.', metavar='DIR', required=True)
@click.option('--outdir', '-o', help='Where to save the results.', metavar='DIR', required=True)
@click.option('--tex_bsdir', help='Where to load the pre-computed texture blend shapes.', metavar='DIR', required=True)
@click.option('--strand_bsdir', help='Where to load the pre-computed strand blend shapes.', metavar='DIR', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', default=None)
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
@click.option('--save_strands', help='Whether to save hair strands colored by reconstruction errors.', metavar='BOOL', type=bool, default=False, show_default=True)
def eval_rawtex_pca(eval_mode, indir, outdir, tex_bsdir, strand_bsdir, seeds, head_mesh, scalp_bounds, save_strands):
    texture_files = sorted(glob.glob(os.path.join(indir, '*.npz')))
    os.makedirs(outdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = np.load(tex_bsdir)
    mean_shape = torch.tensor(data['mean_shape'], dtype=torch.float32, device=device)
    blend_shapes = torch.tensor(data['blend_shapes'], dtype=torch.float32, device=device)
    strand_codec = StrandCodec(model_path=strand_bsdir, num_coeff=10, fft=True).to(device)
    hair_roots = HairRoots(head_mesh=head_mesh, scalp_bounds=scalp_bounds)

    if eval_mode == 'sample':
        data = load_tensor_dict(texture_files[0], device=device)
        roots = data['roots']
        for seed_idx, seed in enumerate(seeds):
            print('Generating guide strands for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            coeff = torch.from_numpy(np.random.RandomState(seed).randn(1, blend_shapes.shape[0])).to(device)
            rawtex = mean_shape + blend(coeff.float(), blend_shapes)
            strand_coeff = rawtex.permute(0, 2, 3, 1).reshape(-1, strand_codec.num_coeff)
            position = strand_codec.decode(strand_coeff)
            position = position.reshape(roots.shape[0], -1, 3)
            position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)
            position = position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
            save_hair(os.path.join(outdir, f'{seed:04d}_guide.data'), c2c(position))
    else:
        metrics = dict(pos_diff=[], cur_diff=[])
        for f in tqdm(texture_files):
            filename = os.path.splitext(os.path.split(f)[1])[0]
            data = load_tensor_dict(f, device=device)
            texture = data['texture'].unsqueeze(0)
            roots = data['roots']

            # encode & decode from pre-computed texture blend shapes
            coeff = project(texture - mean_shape, blend_shapes)
            recon = mean_shape + blend(coeff, blend_shapes)

            # decode from textures to obtain strand geometry
            strand_coeff = recon.permute(0, 2, 3, 1).reshape(-1, strand_codec.num_coeff)
            position = strand_codec.decode(strand_coeff)
            position = position.reshape(roots.shape[0], -1, 3)
            position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)

            gt_strand_coeff = texture.permute(0, 2, 3, 1).reshape(-1, strand_codec.num_coeff)
            gt_position = strand_codec.decode(gt_strand_coeff)
            gt_position = gt_position.reshape(roots.shape[0], -1, 3)
            gt_position = F.pad(gt_position, (0, 0, 1, 0), mode='constant', value=0)

            pos_diff = torch.norm(gt_position - position, dim=-1)
            metrics['pos_diff'].append(c2c(pos_diff.mean()))
            cur_diff = (curvature(gt_position) - curvature(position)).abs()
            metrics['cur_diff'].append(c2c(cur_diff.mean()))

            if save_strands:
                position = position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
                save_hair(os.path.join(outdir, f'{filename}_guide.data'), c2c(position))
                gt_position = gt_position + hair_roots.spherical_to_cartesian(roots).unsqueeze(1)
                save_hair(os.path.join(outdir, f'{filename}_guide_gt.data'), c2c(gt_position))
                write_texture(os.path.join(outdir, f'{filename}_guide.png'), recon[0].permute(1, 2, 0))
                write_texture(os.path.join(outdir, f'{filename}_guide_gt.png'), texture[0].permute(1, 2, 0))

        df = pd.DataFrame.from_dict(metrics)
        bs_fname = os.path.splitext(os.path.basename(tex_bsdir))[0]
        df.to_csv(os.path.join(outdir, f'{bs_fname}.csv'))


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    eval_rawtex_pca()  # pylint: disable=no-value-for-parameter
