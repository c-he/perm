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
@click.option('--hair1', help='Where to load fitted parameters for the first hairstyle', required=True)
@click.option('--hair2', help='Where to load fitted parameters for the second hairstyle', required=True)
@click.option('--steps', type=int, help='Number of steps to interpolate between parameters', default=10)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--head_mesh', help='Head mesh to place hair models', metavar='DIR', type=str)
@click.option('--scalp_bounds', help='Bounding box of the scalp area', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default=[0.1870, 0.8018, 0.4011, 0.8047], show_default=True)
@click.option('--interp_mode', help='Interpolate mode', metavar='STR', type=click.Choice(['theta', 'beta', 'full']), default='theta', show_default=True)
def style_mixing(
    model_path: str,
    hair1: str,
    hair2: str,
    steps: int,
    outdir: str,
    head_mesh: str,
    scalp_bounds: List[float],
    interp_mode: str,
):
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')
    hair_model = Perm(model_path=model_path, head_mesh=head_mesh, scalp_bounds=scalp_bounds).eval().requires_grad_(False).to(device)

    data = np.load(hair1)
    roots = torch.tensor(data['roots'], dtype=torch.float32, device=device)
    theta1 = torch.tensor(data['theta'], dtype=torch.float32, device=device)
    beta1 = torch.tensor(data['beta'], dtype=torch.float32, device=device)

    # data = np.load('fitting/usc-hair/npz/strands00035.npz')
    # beta1 = torch.tensor(data['beta'], dtype=torch.float32, device=device)
    # beta1 = beta1 * 1.3

    data = np.load(hair2)
    theta2 = torch.tensor(data['theta'], dtype=torch.float32, device=device)
    beta2 = torch.tensor(data['beta'], dtype=torch.float32, device=device)

    batch_size = steps + 2
    roots = roots.repeat(batch_size, 1, 1)
    if interp_mode == 'theta':
        # theta2[7:] = theta1[7:].clone()
        theta = [torch.lerp(theta1, theta2, i / (steps + 1)) for i in range(steps + 2)]
        theta = torch.stack(theta)
        beta = beta1.unsqueeze(0).repeat(batch_size, 1, 1)
    elif interp_mode == 'beta':
        beta = [torch.lerp(beta1, beta2, i * 1.3 / (steps + 1)) for i in range(steps + 2)]
        beta = torch.stack(beta)
        theta = theta1.unsqueeze(0).repeat(batch_size, 1, 1)
    else:
        theta = [torch.lerp(theta1, theta2, i / (steps + 1)) for i in range(steps + 2)]
        theta = torch.stack(theta)
        beta = [torch.lerp(beta1, beta2, i / (steps + 1)) for i in range(steps + 2)]
        beta = torch.stack(beta)

    with torch.no_grad():
        for i in range(batch_size):
            out = hair_model(roots=roots[i:i + 1], theta=theta[i:i + 1], beta=beta[i:i + 1])
            save_hair(os.path.join(outdir, f'{filename(hair1)}_{filename(hair2)}_frame{i}.data'), c2c(out['strands'][0].position))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    style_mixing()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
