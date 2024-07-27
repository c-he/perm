import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dnnlib
import legacy
from hair import HairRoots


class Perm(nn.Module):
    """ A wrapper class for 3D Parametric Hair Model.
    """

    def __init__(
        self,
        model_path: str,
        head_mesh: str,
        scalp_bounds: Optional[Tuple[float]] = None,
    ):
        """ ParaHair model constructor.

        Args:
            model_path (str): Path to the folder where all model components are stored.
            head_mesh (str): Path to the head mesh.
            scalp_bounds (Tuple[float]): Precomputed scalp AABB. (default = None)
        """
        super().__init__()

        network_pkl_raw = os.path.join(model_path, 'stylegan2-raw-texture.pkl')
        print('Loading guide texture network from "%s"...' % network_pkl_raw)
        with dnnlib.util.open_url(network_pkl_raw) as f:
            self.G_raw = legacy.load_network_pkl(f)['G_ema'].eval().requires_grad_(False)  # type: ignore

        network_pkl_superres = os.path.join(model_path, 'unet-superres.pkl')
        print('Loading super resolution network from "%s"...' % network_pkl_superres)
        with dnnlib.util.open_url(network_pkl_superres) as f:
            self.G_superres = legacy.load_network_pkl(f)['G'].eval().requires_grad_(False)  # type: ignore

        network_pkl_res = os.path.join(model_path, 'vae-res-texture.pkl')
        print('Loading residual texture network from "%s"...' % network_pkl_res)
        with dnnlib.util.open_url(network_pkl_res) as f:
            self.G_res = legacy.load_network_pkl(f)['G'].eval().requires_grad_(False)  # type: ignore

        self.hair_roots = HairRoots(head_mesh, scalp_bounds)
        u, v = torch.meshgrid(torch.linspace(0, 1, steps=self.G_superres.img_resolution), torch.linspace(0, 1, steps=self.G_superres.img_resolution), indexing='ij')
        uv = torch.dstack((u, v)).permute(2, 1, 0)  # (2, H, W)
        uv_guide = F.interpolate(uv.unsqueeze(0), size=(self.G_raw.img_resolution, self.G_raw.img_resolution), mode='nearest')[0]
        uv_guide = uv_guide.permute(1, 2, 0).reshape(-1, 2)
        uv_guide = self.hair_roots.rescale(uv_guide, inverse=True)
        guide_roots = self.hair_roots.spherical_to_cartesian(uv_guide).unsqueeze(0)
        self.register_buffer('guide_roots', guide_roots)

    @property
    def device(self):
        return self.guide_roots.device

    @property
    def theta_num_ws(self):
        return self.G_raw.backbone.mapping.num_ws

    @property
    def beta_num_ws(self):
        return self.G_res.backbone.num_ws

    def theta_avg(self):
        """ Return midpoint of theta in W space. """
        w_avg = self.G_raw.backbone.mapping.w_avg[None, None, :]  # [1, 1, C]
        return w_avg.repeat(1, self.theta_num_ws, 1)  # [1, L, C]

    def beta_avg(self):
        """ Return midpoint of beta in W space. (theoretically it should be close to 0) """
        w_avg = self.G_res.encoder.w_avg[None, None, :]  # [1, 1, C]
        return w_avg.repeat(1, self.beta_num_ws, 1)  # [1, L, C]

    def guide_strands(
        self,
        theta: torch.Tensor,
        trunc: float = 1.0,
        trunc_cutoff: Optional[int] = None,
    ):
        """ Synthesize guide strands from theta.

        Args:
            theta (torch.Tensor): Parameters to synthesize guide strands.
            trunc (float): Strength of truncation trick. (default = 1.0)
            trunc_cutoff (Optional[int]): Cutoff index of truncation trick. (default = None)

        Returns:
            Dict: Dict of synthesized 32x32 textures/masks and decoded guide strand geometry.
        """
        if theta.ndim == 2:  # synthesis from Z
            out = self.G_raw(theta, truncation_psi=trunc, truncation_cutoff=trunc_cutoff, noise_mode='const')
        else:  # synthesis from W+
            out = self.G_raw.synthesis(theta, noise_mode='const')
        image = out['image']
        image_mask = out['image_mask']
        guide_strands = self.G_raw.sample(image)
        guide_strands.position = guide_strands.position + self.guide_roots.unsqueeze(2)
        return {'image': image, 'image_mask': image_mask, 'strands': guide_strands}

    def superresolution(
        self,
        roots: torch.Tensor,
        img: dict,
        beta: torch.Tensor,
    ):
        """ Upsample guide strands to the final hair model, under the condition of beta.

        Args:
            roots (torch.Tensor): 3D roots to place each hair strand.
            img (Dict): Dict of synthesized 32x32 textures and masks.
            beta (torch.Tensor): Parameters to upsample guide strands to the final hair model.

        Returns:
            Dict: Dict of upsampled 256x256 textures and decoded strand geometry.
        """
        if beta.ndim == 2:  # beta is in W rather than W+ space
            beta = beta.unsqueeze(1).repeat(1, self.G_res.num_ws, 1)
        low_rank_coeff = self.G_superres(img)['image']
        high_rank_coeff = self.G_res.synthesis(beta, noise_mode='const')['image']
        image = torch.cat([low_rank_coeff, high_rank_coeff], dim=1)

        coords = self.hair_roots.cartesian_to_spherical(roots)[..., :2]
        coords = self.hair_roots.rescale(coords)
        strands = self.G_res.sample(image, coords, mode='nearest')
        strands.position = strands.position + roots.unsqueeze(2)
        return {'image': image, 'strands': strands}

    def forward(
        self,
        roots: torch.Tensor,
        theta: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        trunc: float = 1.0,
        trunc_cutoff: Optional[int] = None,
        random_seed: int = 0,
    ):
        """ Forward pass for ParaHair.

        Args:
            roots (torch.Tensor): 3D roots of each hair strand. Required to synthesize hair strands.
            theta (torch.Tensor): Parameters to synthesize guide strands. (default = None)
            beta (torch.Tensor): Parameters to upsample guide strands to the final hair model. (default = None)
            trunc (float): Strength of truncation trick. (default = 1.0)
            trunc_cutoff (Optional[int]): Cutoff index of truncation trick. (default = None)
            random_seed (int): Random seed to use for RNG. (default = 0)
        """
        batch_size = roots.shape[0]
        device = roots.device
        theta = theta if theta is not None else torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, self.G_raw.z_dim)).to(device)
        beta = beta if beta is not None else torch.from_numpy(np.random.RandomState(random_seed).randn(batch_size, self.G_res.w_dim)).to(device)

        out_guide = self.guide_strands(theta, trunc=trunc, trunc_cutoff=trunc_cutoff)
        out = self.superresolution(roots, img={'image_raw': out_guide['image'], 'image_mask': out_guide['image_mask']}, beta=beta)
        out.update(guide_strands=out_guide['strands'], theta=theta, beta=beta)
        return out
