from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from utils.misc import copy2cpu as c2c


def split_patches(data: Union[np.ndarray, torch.Tensor], patch_size: int, mask: Optional[Union[np.ndarray, torch.Tensor]] = None, overlap: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """ Split texture data into uniform grids of patches.

    Args:
        data (Union[np.ndarry, torch.Tensor]): Texture data of shape (batch_size, feature_dim, texture_size, texture_size).
        patch_size (int): Size of the patches.
        mask (Union[np.ndarry, torch.Tensor]): Mask data of shape (batch_size, 1, texture_size, texture_size). If applied, split patches only in the unmasked area. (default = None)
        overlap (bool): If true, split patches with overlap. (default = True)

    Returns:
        Union[np.ndarray, torch.Tensor]: Patches of shape (num_patches, batch_size, feature_dim, patch_size, patch_size).
    """
    batch_size, feature_dim = data.shape[:2]
    np_conversion = False
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)
        np_conversion = True

    if overlap:
        patches = data.unfold(2, patch_size, patch_size // 2).unfold(3, patch_size, patch_size // 2)
    else:
        patches = data.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(2, 3, 0, 1, 4, 5).reshape(-1, batch_size, feature_dim, patch_size, patch_size)

    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.int32)

        if overlap:
            mask = mask.unfold(2, patch_size, patch_size // 2).unfold(3, patch_size, patch_size // 2)
        else:
            mask = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        mask = mask.permute(2, 3, 0, 1, 4, 5).reshape(-1, batch_size * patch_size * patch_size).sum(dim=-1)  # (num_patches, )
        patches = patches.index_select(dim=0, index=torch.nonzero(mask, as_tuple=True)[0])

    if np_conversion:
        patches = c2c(patches)

    return patches


def assemble_patches(patches: torch.Tensor, overlap: bool = True) -> torch.Tensor:
    """ Assemble patches into a unified texture.

    Args:
        patches (torch.Tensor): List of patches of shape (num_patches, batch_size, feature_dim, patch_size, patch_size).
        overlap (bool): If true, assemble overlapping patches and averge them. (default = True)

    Returns:
        (torch.Tensor): Assembled textures of shape (batch_size, feature_dim, height, width).
    """
    num_patches, batch_size, feature_dim, patch_size = patches.shape[:4]
    num_patches_sqrt = int(np.sqrt(num_patches))

    if overlap:
        output_size = int((num_patches_sqrt + 1) / 2 * patch_size)
        patches = patches.permute(1, 2, 3, 4, 0)
        patches = patches.contiguous().view(batch_size, -1, num_patches)
        weight = torch.ones_like(patches)
        patches = F.fold(patches, output_size=(output_size, output_size), kernel_size=patch_size, stride=patch_size // 2)
        weight = F.fold(weight, output_size=(output_size, output_size), kernel_size=patch_size, stride=patch_size // 2)
        output = patches / weight
    else:
        output_size = num_patches_sqrt * patch_size
        output = (patches.permute(1, 2, 0, 3, 4)
                  .reshape(batch_size, feature_dim, num_patches_sqrt, num_patches_sqrt, patch_size, patch_size)
                  .transpose(3, 4)
                  .reshape(batch_size, feature_dim, output_size, output_size)
                  )

    return output


""" Inefficient implementations with Python loops rather than PyTorch Ops
"""

# def split_patches(data: Union[np.ndarray, torch.Tensor], patch_size: int, overlap: bool = True) -> Union[np.ndarray, torch.Tensor]:
#     """ Split texture data into uniform grids of patches.

#     Args:
#         data (Union[np.ndarry, torch.Tensor]): Texture data of shape (batch_size, feature_dim, texture_size, texture_size).
#         patch_size (int): Size of the patches.
#         overlap (bool): If true, split patches with overlap. (default = True)

#     Returns:
#         Union[np.ndarray, torch.Tensor]: Patches of shape (batch_size x N^2, feature_dim, patch_size, patch_size).
#     """
#     height, width = data.shape[-2:]
#     patches = []

#     for i in range(0, height - patch_size + 1, patch_size // 2 if overlap else patch_size):
#         for j in range(0, width - patch_size + 1, patch_size // 2 if overlap else patch_size):
#             patches.append(data[:, :, i:i + patch_size, j:j + patch_size])

#     if isinstance(data, np.ndarray):
#         patches = np.concatenate(patches)
#     else:
#         patches = torch.cat(patches)

#     return patches


# def assemble_patches(patches: torch.Tensor, overlap: bool = True) -> torch.Tensor:
#     """ Assemble patches into a unified texture.

#     Args:
#         patches (torch.Tensor): List of patches of shape (num_patches, batch_size, feature_dim, patch_size, patch_size).
#         overlap (bool): Whether to overlap patches when assembling them. (default = True)

#     Returns:
#         (torch.Tensor): Assembled textures of shape (batch_size, feature_dim, height, width).
#     """
#     num_patches = patches.shape[0]
#     patch_size = patches.shape[-1]
#     if overlap:
#         output_size = (np.sqrt(num_patches) + 1) / 2 * patch_size
#     else:
#         output_size = np.sqrt(num_patches) * patch_size
#     output_size = int(output_size)
#     output = torch.zeros(*patches.shape[1:3], output_size, output_size, device=patches.device)
#     weight = torch.zeros(patches.shape[1], 1, output_size, output_size, device=patches.device)

#     index = 0
#     for i in range(0, output_size - patch_size + 1, patch_size // 2 if overlap else patch_size):
#         for j in range(0, output_size - patch_size + 1, patch_size // 2 if overlap else patch_size):
#             output[:, :, i:i + patch_size, j:j + patch_size] += patches[index]
#             weight[:, :, i:i + patch_size, j:j + patch_size] += 1.0
#             index += 1

#     output = output / weight

#     return output
