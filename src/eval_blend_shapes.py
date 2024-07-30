import glob
import os

import click
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from hair import load_hair, save_hair
from models import StrandCodec
from utils.metric import curvature
from utils.misc import copy2cpu as c2c

# ----------------------------------------------------------------------------


@click.command()
# Required.
@click.option('--indir', '-i', help='Where to load the data.', metavar='DIR', required=True)
@click.option('--outdir', '-o', help='Where to save the results.', metavar='DIR', required=True)
@click.option('--bsdir', help='Where to load the pre-computed blend shapes.', metavar='DIR', required=True)
@click.option('--n_coeff', help='Number of PCA coefficients.', metavar='INT', type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--save_strands', help='Whether to save hair strands colored by reconstruction errors.', metavar='BOOL', type=bool, default=False, show_default=True)
def eval_blend_shapes(indir, outdir, bsdir, n_coeff, save_strands):
    hair_files = sorted(glob.glob(os.path.join(indir, '*.data')))
    os.makedirs(outdir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    strand_codec = StrandCodec(bsdir, n_coeff, fft=True if 'fft' in bsdir else False)
    strand_codec = strand_codec.to(device)

    metrics = dict(pos_diff=[], cur_diff=[])
    for f in tqdm(hair_files):
        filename = os.path.splitext(os.path.split(f)[1])[0]
        data = torch.tensor(load_hair(f), dtype=torch.float32, device=device)
        roots = data[:, 0:1].clone()

        gt_position = data - roots
        print(gt_position.shape)
        position = strand_codec(gt_position[:, 1:])
        position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)

        pos_diff = torch.norm(gt_position - position, dim=-1)
        metrics['pos_diff'].append(c2c(pos_diff.mean()))
        cur_diff = (curvature(gt_position) - curvature(position)).abs()
        metrics['cur_diff'].append(c2c(cur_diff.mean()))

        if save_strands:
            position = position + roots
            gt_position = gt_position + roots
            save_hair(os.path.join(outdir, f'{filename}.data'), c2c(position))
            save_hair(os.path.join(outdir, f'{filename}_gt.data'), c2c(gt_position))

    df = pd.DataFrame.from_dict(metrics)
    bs_fname = os.path.splitext(os.path.basename(bsdir))[0]
    df.to_csv(os.path.join(outdir, f'{bs_fname}_metrics_{n_coeff}_pc.csv'))


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    eval_blend_shapes()  # pylint: disable=no-value-for-parameter
