import os
import re
from typing import List, Union

import click
import numpy as np
import torch

from hair import save_hair
from hair.hair_models import Perm
from utils.misc import copy2cpu as c2c
from utils.misc import filename

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
@click.option('--params', help='Where to load fitted parameters', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
def gen_from_parameters(
    model_path: str,
    params: str,
    outdir: str,
    head_mesh: str,
    scalp_bounds: List[float],
):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')
    hair_model = Perm(model_path=model_path, head_mesh=head_mesh, scalp_bounds=scalp_bounds).eval().requires_grad_(False).to(device)
    fname = filename(params)
    data = np.load(params)
    roots = torch.tensor(data['roots'], dtype=torch.float32, device=device)
    theta = torch.tensor(data['theta'], dtype=torch.float32, device=device)
    beta = torch.tensor(data['beta'], dtype=torch.float32, device=device)
    print(f'roots: {roots.shape}')
    print(f'theta: {theta.shape}')
    print(f'beta: {beta.shape}')

    out = hair_model(roots=roots.unsqueeze(0), theta=theta.unsqueeze(0), beta=beta.unsqueeze(0))
    save_hair(os.path.join(outdir, f'{fname}.data'), c2c(out['strands'][0].position))
    save_hair(os.path.join(outdir, f'{fname}_guide.data'), c2c(out['guide_strands'][0].position))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    gen_from_parameters()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
