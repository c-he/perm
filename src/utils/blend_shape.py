import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA, SparsePCA


def solve_blend_shapes(data: np.ndarray, num_blend_shapes: int, svd_solver: str) -> np.ndarray:
    """ Solve principal components/blend shapes for 3D hair.

    Args:
        data (np.ndarray): Hair data of shape (num_data, ...).
        num_blend_shapes (int): Number of blend shapes to solve.
        svd_solver (str): SVD solver used for PCA.

    Returns:
        Tuple[np.ndarray]: Blend shapes of shape (num_blend_shapes, ...), and variance ratio explained by each blend shape.
    """
    shape = data.shape[1:]
    pca = PCA(n_components=num_blend_shapes, whiten=True, svd_solver=svd_solver)
    pca.fit(data.reshape(data.shape[0], -1))
    blend_shapes = pca.components_.reshape(-1, *shape)

    return blend_shapes, pca.explained_variance_ratio_

# def solve_blend_shapes(data: np.ndarray, num_blend_shapes: int, svd_solver: str) -> np.ndarray:
#     """ Solve principal components/blend shapes for 3D hair.

#     Args:
#         data (np.ndarray): Hair data of shape (num_data, ...).
#         num_blend_shapes (int): Number of blend shapes to solve.
#         svd_solver (str): SVD solver used for PCA.

#     Returns:
#         Tuple[np.ndarray]: Blend shapes of shape (num_blend_shapes, ...), and variance ratio explained by each blend shape.
#     """
#     shape = data.shape[1:]
#     pca = SparsePCA(n_components=num_blend_shapes, method='lars', n_jobs=-1)
#     pca.fit(data.reshape(data.shape[0], -1))
#     blend_shapes = pca.components_.reshape(-1, *shape)

#     return blend_shapes


def project(data: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Project data to the subspace spanned by bases.

    Args:
        data (torch.Tensor): Hair data of shape (batch_size, ...).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Projected parameters of shape (batch_size, num_blend_shapes).
    """
    return torch.einsum('bn,cn->bc', data.flatten(start_dim=1), basis.flatten(start_dim=1))


def blend(coeff: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """ Blend parameters and the corresponding blend shapes.

    Args:
        coeff (torch.Tensor): Parameters (blend shape coefficients) of shape (batch_size, num_blend_shapes).
        basis (torch.Tensor): Blend shapes of shape (num_blend_shapes, ...).

    Returns:
        (torch.Tensor): Blended results of shape (batch_size, ...).
    """
    return torch.einsum('bn,n...->b...', coeff, basis)


def sample(coords: torch.Tensor, blend_shape: torch.Tensor, mode: str = 'bilinear') -> torch.Tensor:
    """ Sample neural blend shapes with given coordinates.

    Args:
        coords (torch.Tensor): Sample coordinates of shape (batch_size, num_coords, 2) in [0, 1] x [0, 1].
        blend_shape (torch.Tensor): Blend shapes of shape (batch_size, feature_dim, height, width).
        mode (str): Interpolation mode for sampling.

    Returns:
        (torch.Tensor): Sampled neural features of shape (batch_size, num_coords, feature_dim).
    """
    grid = coords * 2 - 1  # (batch_size, num_coords, 2), in [-1, 1]
    samples = F.grid_sample(blend_shape, grid.unsqueeze(2), mode=mode, align_corners=True)  # (batch_size, feature_dim, num_coords, 1)
    samples = samples.squeeze(-1).mT  # (batch_size, num_coords, feature_dim)

    return samples
