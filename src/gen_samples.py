import os
import re
from typing import List, Union

import click
import numpy as np
import torch

from hair import save_hair
from hair.hair_models import Perm
from utils.image import write_texture
from utils.misc import copy2cpu as c2c

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
@click.option('--model', 'model_path', help='Pre-trained model directory', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=8, show_default=True)
@click.option('--roots', metavar='DIR', help='Path to the hair root file for resampling.', required=True)
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
def sample_perm(
    model_path: str,
    seeds: List[int],
    outdir: str,
    truncation_psi: float,
    truncation_cutoff: int,
    roots: str,
    head_mesh: str,
    scalp_bounds: List[float],
):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')
    hair_model = Perm(model_path=model_path, head_mesh=head_mesh, scalp_bounds=scalp_bounds).eval().requires_grad_(False).to(device)
    roots, _ = hair_model.hair_roots.load_txt(roots)
    roots = roots.to(device)
    print(f'roots: {roots.shape}')

    for seed_idx, seed in enumerate(seeds):
        print('Generating sample for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        out = hair_model(roots=roots.unsqueeze(0), trunc=truncation_psi, trunc_cutoff=truncation_cutoff, random_seed=seed)
        write_texture(os.path.join(outdir, f'seed{seed:04d}.png'), out['image'][0].permute(1, 2, 0))
        save_hair(os.path.join(outdir, f'seed{seed:04d}.data'), c2c(out['strands'][0].position))
        save_hair(os.path.join(outdir, f'seed{seed:04d}_guide.data'), c2c(out['guide_strands'][0].position))
        np.savez(os.path.join(outdir, f'seed{seed:04d}.npz'), theta=c2c(out['theta'][0]), beta=c2c(out['beta'][0]))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    sample_perm()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
