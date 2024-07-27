from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hair import Strands
from utils.acos import acos_linear_extrapolation
from utils.rotation import rotation_6d_to_matrix


class GeodesicLoss(nn.Module):
    def __init__(self, cos_angle: bool = False, reduction: str = 'mean') -> None:
        super().__init__()
        self.cos_angle = cos_angle
        self.reduction = reduction

    def _compute_geodesic_distance(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        """ Compute the geodesic distance between two rotation matrices.

        Args:
            R1/R2 (torch.Tensor): Two rotation matrices of shape (..., 3, 3).

        Returns:
            (torch.Tensor) The minimal angular difference between two rotation matrices.
        """
        R = torch.matmul(R1, R2.mT)  # (..., 3, 3)
        rot_trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        # phi -- rotation angle
        phi_cos = ((rot_trace - 1.0) * 0.5).clamp(min=-1, max=1)
        if self.cos_angle:
            return 1.0 - phi_cos  # [0, 2]
        else:
            phi = acos_linear_extrapolation(phi_cos)  # [0, pi]
            return phi

    def __call__(self, R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
        loss = self._compute_geodesic_distance(R1, R2)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class StrandGeometricLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, s1, s2):
        loss = dict()
        loss['pos'] = F.l1_loss(s1.position, s2.position)

        d1 = F.normalize(s1.position[..., 1:, :] - s1.position[..., :-1, :], dim=-1)
        d2 = F.normalize(s2.position[..., 1:, :] - s2.position[..., :-1, :], dim=-1)
        loss['rot'] = (1 - F.cosine_similarity(d1, d2, dim=-1)).mean()

        b1 = torch.norm(torch.cross(d1[..., 1:, :], d1[..., :-1, :], dim=-1), dim=-1)
        b2 = torch.norm(torch.cross(d2[..., 1:, :], d2[..., :-1, :], dim=-1), dim=-1)
        loss['cur'] = F.l1_loss(b1, b2)

        return loss
