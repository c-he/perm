import glob
import os

import click
import pandas as pd
import torch
from tqdm import tqdm

from hair import load_hair
from utils.metric import curvature
from utils.misc import copy2cpu as c2c

# ----------------------------------------------------------------------------


@click.command()
# Required.
@click.option('--indir', '-i', help='Where to load the data.', metavar='DIR', required=True)
@click.option('--outdir', '-o', help='Where to save the results.', metavar='DIR', required=True)
def eval_strand_reconstruction(indir, outdir):
    hair_files = sorted(glob.glob(os.path.join(indir, '*[!_gt].data')))
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics = dict(pos_diff=[], cur_diff=[])
    for f in tqdm(hair_files):
        filename = os.path.splitext(os.path.split(f)[1])[0]
        position = torch.tensor(load_hair(f), dtype=torch.float32, device=device)
        gt_position = torch.tensor(load_hair(os.path.join(indir, f'{filename}_gt.data')), dtype=torch.float32, device=device)

        pos_diff = torch.norm(gt_position - position, dim=-1)
        metrics['pos_diff'].append(c2c(pos_diff.mean()))
        cur_diff = (curvature(gt_position) - curvature(position)).abs()
        metrics['cur_diff'].append(c2c(cur_diff.mean()))

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(os.path.join(outdir, 'recon_metrics.csv'))


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    eval_strand_reconstruction()  # pylint: disable=no-value-for-parameter
