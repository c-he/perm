from typing import Tuple

import torch
import torch.nn.functional as F

from utils.acos import acos_linear_extrapolation
from utils.rotation import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from utils.misc import EPSILON


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: return torch.einsum('...n,...n->...', x, y)


def rotation_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """ Compute the rotation matrix `R` between unit vectors `v1` and `v2`, such that `v2 = Rv1`.

    Args:
        v1/v2 (torch.Tensor): 3D unit vectors of shape (..., 3).

    Returns:
        (torch.Tensor): Rotation matrices of shape (..., 3, 3).
    """
    axis = torch.cross(v1, v2, dim=-1)
    axis = F.normalize(axis, dim=-1)
    angle = dot(v1, v2).clamp(min=-1.0, max=1.0)
    # resolve singularity when angle=pi, since both angle 0 and angle pi will produce zero axes
    v_clone = v1[angle == -1]
    axis_ortho = torch.zeros_like(v_clone)
    axis_ortho[..., 0] = v_clone[..., 2] - v_clone[..., 1]
    axis_ortho[..., 1] = v_clone[..., 0] - v_clone[..., 2]
    axis_ortho[..., 2] = v_clone[..., 1] - v_clone[..., 0]
    # if two vectors v1 and v2 point in opposite directions (angle=pi), modify the axis to be orthogonal to v1
    axis[angle == -1] = F.normalize(axis_ortho, dim=-1)
    # angle = acos_linear_extrapolation(angle)  # [0, pi]
    angle = torch.acos(angle.clamp(min=-1.0 + EPSILON, max=1.0 - EPSILON))  # [0, pi]

    axis_angle = axis * angle[..., None]
    rotmat = axis_angle_to_matrix(axis_angle)

    return rotmat


def parallel_transport_rotation(x: torch.Tensor) -> torch.Tensor:
    """ Compute a set of rotation matrices for parallel transport.

    Args:
        x (torch.Tensor): Discrete points of shape (..., num_samples, 3).

    Returns:
        (torch.Tensor): Rotation matrices of shape (..., num_samples - 2, 3, 3).
    """
    t = F.normalize(x[..., 1:, :] - x[..., :-1, :], dim=-1)  # (..., num_samples - 1, 3)
    rotmat = rotation_between_vectors(t[..., :-1, :], t[..., 1:, :])  # (..., num_samples - 2, 3, 3)

    return rotmat


def cartesian_to_rotational_repr(position: torch.Tensor, global_rot: bool = True, return_rot6d: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Compute rotational representation from the Cartesian coordinates.

    Args:
        position (torch.Tensor): Polyline strands of shape (..., num_samples, 3).
        global_rot (bool): If true, return rotation matrices relative to the root. (default = True)
        return_rot6d (bool): If true, return rotation matrices in the 6D form. (default = False)

    Return:
        Tuple[torch.Tensor, torch.Tensor]: Rotation matrices and segment lengths computed from 3D points.
    """
    if global_rot:
        direction = F.normalize(position[..., 1:, :] - position[..., :-1, :], dim=-1)
        forward = torch.zeros_like(direction)
        forward[..., 1] = -1
        rotation = rotation_between_vectors(forward, direction)  # (..., num_samples - 1, 3, 3)
    else:
        dummy = torch.zeros_like(position[..., 0, :])
        dummy[..., 1] = -1
        # create a dummy origin to compute the rotation between (0, -1, 0) and the first segment of each strand
        origin = position[..., 0, :] - dummy
        dummy_x = torch.cat((origin.unsqueeze(-2), position), dim=-2)
        rotation = parallel_transport_rotation(dummy_x)  # (..., num_samples - 1, 3, 3)
    if return_rot6d:
        rotation = matrix_to_rotation_6d(rotation)  # (..., num_samples - 1, 6)

    length = torch.norm(position[..., 1:, :] - position[..., :-1, :], dim=-1)  # (..., num_samples - 1)

    return rotation, length


def integrate_strand_position(segment: torch.Tensor) -> torch.Tensor:
    """ Integrate each segment to obtain the final strand.

    Args:
        segment (torch.Tensor): Polyline segments of shape (..., num_samples - 1, 3).

    Returns:
        (torch.Tensor): 3D point positions of shape (..., num_samples, 3).
    """
    position = torch.zeros_like(segment)
    position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)

    for i in range(1, position.shape[-2]):
        position[..., i, :] = position[..., i - 1, :] + segment[..., i - 1, :]

    return position


def forward_kinematics(rotation: torch.Tensor, length: torch.Tensor, global_rot: bool = True) -> torch.Tensor:
    """ Compute Cartesian coordinates of strands from rotation matrices and segment lengths.

    Args:
        rotation (torch.Tensor): Rotation matrices of shape (..., num_samples - 1, 3, 3) or 6D rotation of shape (..., num_samples - 1, 6).
        length (torch.Tensor): Segment lengths of shape (..., num_samples - 1, 3).
        global_rot (bool): Whether the rotation matrices are in global or relative representation. (default = True)

    Returns:
        (torch.Tensor): 3D point positions of shape (..., num_samples, 3).
    """
    rotmat = rotation.clone()
    if rotmat.shape[-1] == 6:
        rotmat = rotation_6d_to_matrix(rotmat)
    if not global_rot:
        for i in range(1, rotmat.shape[1]):
            rotmat[..., i, :, :] = torch.matmul(rotmat[..., i, :, :].clone(), rotmat[..., i - 1, :, :].clone())
    forward = torch.zeros(rotmat.shape[:-1], device=rotation.device)
    forward[..., 1] = -length
    segment = torch.matmul(rotmat, forward[..., None]).squeeze(-1)

    return integrate_strand_position(segment)


# def build_Bishop_frame(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """ Build twist-free Bishop frames on each edge of the rods.

#     Args:
#         x (torch.Tensor): Discrete rods. (batch_size, num_samples, 3)

#     Returns:
#         Tuple[torch.Tensor, torch.Tensor, torch.Tensor] Bishop frames {t, u, v}.
#     """
#     t = F.normalize(x[:, 1:] - x[:, :-1], dim=-1)
#     u = torch.zeros_like(t)

#     # construct u0 that is perpendicular to t0
#     u[:, 0, 0] = t[:, 0, 2] - t[:, 0, 1]
#     u[:, 0, 1] = t[:, 0, 0] - t[:, 0, 2]
#     u[:, 0, 2] = t[:, 0, 1] - t[:, 0, 0]

#     # construct rotations for discrete parallel transport
#     P = parallel_transport_rotation(x, global_rot=False)

#     # discrete parallel transport
#     for i in range(1, t.shape[1]):
#         u[:, i] = torch.matmul(P[:, i - 1], u[:, i - 1, :, None]).squeeze(-1)
#     u = F.normalize(u, dim=-1)
#     v = torch.cross(t, u, dim=-1)
#     v = F.normalize(v, dim=-1)

#     return t, u, v


# def compute_material_curvatures(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#     """ Compute material curvatures on each edge of the rods.

#     Args:
#         x (torch.Tensor): discrete rods. (batch_size, num_samples, 3)

#     Returns:
#        Tuple[torch.Tensor, torch.Tensor] material curvatures (w1, w2). (batch_size, num_samples - 2)
#     """
#     e = x[:, 1:] - x[:, :-1]
#     kb = 2 * torch.cross(e[:, :-1], e[:, 1:], dim=-1)
#     denom = torch.norm(e[:, :-1], dim=-1) * torch.norm(e[:, 1:], dim=-1) + dot(e[:, :-1], e[:, 1:])
#     denom = denom.clamp_min(1e-12).unsqueeze(-1)
#     kb /= denom

#     t, u, v = build_Bishop_frame(x)

#     w1 = dot(kb, u[:, :-1])
#     w2 = dot(kb, v[:, :-1])

#     return w1, w2


# def recover_discrete_rods(x0: torch.Tensor, t0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
#     """ Recover the discrete rods from their curvature representation.

#     Args:
#         x0 (torch.Tensor): the first point on the rod. (batch_size, 3)
#         t0 (torch.Tensor): the first (normalized) edge of the rod. (batch_size, 3)
#         w1, w2 (torch.Tensor): material curvatures. (batch_size, num_samples - 2)
#         l (torch.Tensor): length of the edges. (batch_size, num_samples - 1)

#     Returns:
#         (torch.Tensor) discrete rods. (batch_size, num_samples, 3)
#     """
#     x = x0[:, None, :].repeat(1, l.shape[1] + 1, 1)  # (batch_size, num_samples, 3)
#     t = t0[:, None, :].repeat(1, l.shape[1], 1)  # (batch_size, num_samples - 1, 3)

#     u = torch.zeros_like(t)
#     v = torch.zeros_like(t)
#     # build Bishop frame on the first edge
#     u[:, 0, 0] = t[:, 0, 2] - t[:, 0, 1]
#     u[:, 0, 1] = t[:, 0, 0] - t[:, 0, 2]
#     u[:, 0, 2] = t[:, 0, 1] - t[:, 0, 0]
#     u[:, 0] = F.normalize(u[:, 0], dim=-1)
#     v[:, 0] = F.normalize(torch.cross(t[:, 0], u[:, 0], dim=-1), dim=-1)

#     for i in range(1, x.shape[1]):
#         # update x_i
#         x[:, i] = x[:, i - 1] + t[:, i - 1].clone() * l[:, i - 1, None]
#         if i < x.shape[1] - 1:
#             # compute kb
#             kb = u[:, i - 1].clone() * w1[:, i - 1, None] + v[:, i - 1].clone() * w2[:, i - 1, None]
#             # compute rotation P
#             axis = F.normalize(kb, dim=-1)
#             angle = 2.0 * torch.atan(torch.norm(kb, dim=-1) / 2)
#             P = axis * angle[..., None]
#             P = axis_angle_to_matrix(P)
#             # update t^i, u^i, v^i
#             t[:, i] = torch.matmul(P, t[:, i - 1, :, None].clone()).squeeze(-1)
#             t[:, i] = F.normalize(t[:, i].detach(), dim=-1)
#             u[:, i] = torch.matmul(P, u[:, i - 1, :, None].clone()).squeeze(-1)
#             u[:, i] = F.normalize(u[:, i].detach(), dim=-1)
#             v[:, i] = F.normalize(torch.cross(t[:, i].clone(), u[:, i].clone(), dim=-1), dim=-1)

#     return x
