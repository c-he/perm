from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from hair.rotational_repr import cartesian_to_rotational_repr, dot, forward_kinematics, integrate_strand_position, rotation_between_vectors
from utils.misc import EPSILON
from utils.rotation import rotation_6d_to_matrix


@dataclass
class Strands:
    position: Optional[torch.Tensor] = None
    """ 3D position on each point of the strand """

    rotation: Optional[torch.Tensor] = None
    """ Global rotation on each edge of the strand (either 6D rotation or rotation matrix) """

    length: Optional[torch.Tensor] = None
    """ Length on each edge of the strand """

    def __len__(self) -> int:
        """
        Returns:
            (int): Number of strands in the pack.
        """
        for f in fields(self):
            attr = getattr(self, f.name)
            if attr is not None:
                return attr.shape[0]

        raise AttributeError('Empty Strands class does not have attribute `__len__`')

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns:
            (int): Shape of the strands pack.
        """
        if self.position is not None:
            return self.position.shape[:-2]
        if self.rotation is not None:
            return self.rotation.shape[:-2] if self.rotation.shape[-1] == 6 else self.rotation.shape[:-3]
        if self.length is not None:
            return self.length.shape[:-1]
        else:
            raise AttributeError('Empty Strands class does not have attribute `shape`')

    @property
    def ndim(self) -> int:
        """
        Returns:
            (int): Number of spatial dimensions for the strands pack.
        """
        if self.position is not None:
            return self.position.ndim - 2
        if self.rotation is not None:
            return self.rotation.ndim - 2 if self.rotation.shape[-1] == 6 else self.rotation.ndim - 3
        if self.length is not None:
            return self.length.ndim - 1
        else:
            raise AttributeError('Empty Strands class does not have attribute `ndim`')

    def _apply(self, fn) -> Strands:
        """ Apply the function `fn` on each of the channels, if not None.
            Returns a new instance with the processed channels.
        """
        data = {}
        for f in fields(self):
            attr = getattr(self, f.name)
            data[f.name] = None if attr is None else fn(attr)
        return Strands(**data)

    @staticmethod
    def _apply_on_list(lst, fn) -> Strands:
        """ Applies the function `fn` on each entry in the list `lst`.
            Returns a new instance with the processed channels.
        """
        data = {}
        for l in lst:  # gather the property of all entries into a dict
            for f in fields(l):
                attr = getattr(l, f.name)
                if f.name not in data:
                    data[f.name] = []
                data[f.name].append(attr)
        for k, v in data.items():
            data[k] = None if None in v else fn(v)
        return Strands(**data)

    @classmethod
    def cat(cls, lst: List[Strands], dim: int = 0) -> Strands:
        """ Concatenate multiple strands into a single Strands object.

        Args:
            lst (List[Strands]): List of strands to concatenate, expected to have the same spatial dimensions, except of dimension dim.
            dim (int): Spatial dimension along which the concatenation should take place.

        Returns:
            (Strands): A single Strands object with the concatenation of given strands packs.
        """
        if dim < 0:
            dim -= 1
        if dim > lst[0].ndim - 1 or dim < -lst[0].ndim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{-lst[0].ndim}, {lst[0].ndim-1}], but got {dim})")

        return Strands._apply_on_list(lst, lambda x: torch.cat(x, dim=dim))

    @classmethod
    def stack(cls, lst: List[Strands], dim: int = 0) -> Strands:
        """ Stack multiple strands into a single Strands object.

        Args:
            lst (List[Strands]): List of strands to stack, expected to have the same spatial dimensions.
            dim (int): Spatial dimension along which the stack should take place.

        Returns:
            (Strands): A single Strands object with the stacked strands packs.
        """
        return Strands._apply_on_list(lst, lambda x: torch.stack(x, dim=dim))

    def __getitem__(self, idx) -> Strands:
        """ Get strand on the index `idx`. """
        return self._apply(lambda x: x[idx])

    def reshape(self, *dims: Tuple[int]) -> Strands:
        """ Reshape strands to the given `dims`. """
        def _reshape(x):
            extra_dims = self.ndim - x.ndim
            return x.reshape(*dims + x.shape[extra_dims:])
        return self._apply(_reshape)

    def index_select(self, dim: int, index: torch.Tensor) -> Strands:
        """ Index strands along dimension `dim` using the entries in `index`. """
        return self._apply(lambda x: x.index_select(dim, index))

    def squeeze(self, dim: int) -> Strands:
        """ Squeeze strands on dimension `dim`. """
        return self._apply(lambda x: x.squeeze(dim))

    def unsqueeze(self, dim: int) -> Strands:
        """ Unsqueeze strands on dimension `dim`. """
        return self._apply(lambda x: x.unsqueeze(dim))

    def contiguous(self) -> Strands:
        """ Force strands to reside on a contiguous memory. """
        return self._apply(lambda x: x.contiguous())

    def to(self, *args, **kwargs) -> Strands:
        """ Shift strands to a different device / dtype. """
        return self._apply(lambda x: x.to(*args, **kwargs))

    @staticmethod
    def _low_pass_filter(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """ Apply low-pass filtering on the input (implemented as 1D convolution with moving average kernels). """
        half = int(kernel_size // 2)
        x_conv = x.clone()

        start_idx = 1
        for i in range(start_idx, x_conv.shape[-2] - 1):
            i0, i1 = i - half, i + half
            window = torch.zeros_like(x[..., i, :])
            for j in range(i0, i1 + 1):
                if j < 0:
                    p = 2.0 * x[..., 0, :] - x[..., -j, :]
                elif j >= x.shape[-2]:
                    p = 2.0 * x[..., -1, :] - x[..., x.shape[-2] - j - 2, :]
                else:
                    p = x[..., j, :]
                window += p
            x_conv[..., i, :] = window / kernel_size

        return x_conv

    def smooth(self, kernel_size: int) -> Strands:
        """ Smooth strands with a low-pass filter. """
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1
        position = Strands._low_pass_filter(self.position, kernel_size)

        return Strands(position=position)

    def filter(self, index: torch.Tensor) -> Strands:
        """ Filter strands according to the given index. """
        position = self.position.clone()
        position[index] = 0

        return Strands(position=position)

    def to_canonical(self, guide_strands: Strands) -> Strands:
        """ Transform strands into canonical space (transformation $\phi$ in the paper). """
        if guide_strands.rotation is None or guide_strands.length is None:
            rotation, length = cartesian_to_rotational_repr(guide_strands.position, global_rot=True)
            guide_strands.rotation = rotation
            guide_strands.length = length

        if guide_strands.rotation.shape[-1] == 6:
            guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

        segment = self.position[..., 1:, :] - self.position[..., :-1, :]
        segment = torch.matmul(torch.linalg.inv(guide_strands.rotation), segment[..., None]).squeeze(-1)  # eliminate rotation from guide strands
        forward = torch.zeros_like(segment)
        forward[..., 1] = -1

        position = integrate_strand_position(segment)
        if self.rotation is None:
            rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
        else:
            if self.rotation.shape[-1] == 6:
                self.rotation = rotation_6d_to_matrix(self.rotation)
            rotation = torch.matmul(torch.linalg.inv(guide_strands.rotation), self.rotation)
        if self.length is None:
            length = torch.norm(segment, dim=-1)
        else:
            length = self.length

        return Strands(position=position,
                       rotation=rotation,
                       length=length
                       )

    def to_world(self, guide_strands: Strands) -> Strands:
        """ Transform strands into world space (transformation $\phi^{-1}$ in the paper). """
        if guide_strands.rotation is None or guide_strands.length is None:
            rotation, length = cartesian_to_rotational_repr(guide_strands.position, global_rot=True)
            guide_strands.rotation = rotation
            guide_strands.length = length

        if guide_strands.rotation.shape[-1] == 6:
            guide_strands.rotation = rotation_6d_to_matrix(guide_strands.rotation)

        segment = self.position[..., 1:, :] - self.position[..., :-1, :]
        segment = torch.matmul(guide_strands.rotation, segment[..., None]).squeeze(-1)
        forward = torch.zeros_like(segment)
        forward[..., 1] = -1

        position = integrate_strand_position(segment)
        if self.rotation is None:
            rotation = rotation_between_vectors(forward, F.normalize(segment, dim=-1))
        else:
            if self.rotation.shape[-1] == 6:
                self.rotation = rotation_6d_to_matrix(self.rotation)
            rotation = torch.matmul(guide_strands.rotation, self.rotation)
        if self.length is None:
            length = torch.norm(segment, dim=-1)
        else:
            length = self.length

        return Strands(position=position,
                       rotation=rotation,
                       length=length
                       )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, strand_repr: str, global_rot: bool = True) -> Strands:
        """ Create a Strands object from torch Tensor with the given representation. """
        if strand_repr == 'rotation':  # 6d rotation + length
            rotation = tensor[..., :6]
            length = torch.abs(tensor[..., 6])
            position = forward_kinematics(rotation, length, global_rot)
            return Strands(position=position, rotation=rotation, length=length)
        elif strand_repr == 'direction':  # direction (finite difference of position)
            position = integrate_strand_position(tensor)
            forward = torch.zeros_like(tensor)
            forward[..., 1] = -1
            rotation = rotation_between_vectors(forward, F.normalize(tensor, dim=-1))
            length = torch.norm(tensor, dim=-1)
            return Strands(position=position, rotation=rotation, length=length)
        elif strand_repr == 'position':  # position
            position = F.pad(tensor, (0, 0, 1, 0), mode='constant', value=0)
            direction = F.normalize(position[..., 1:, :] - position[..., :-1, :], dim=-1)
            forward = torch.zeros_like(direction)
            forward[..., 1] = -1
            rotation = rotation_between_vectors(forward, direction)
            length = torch.norm(position[..., 1:, :] - position[..., :-1, :], dim=-1)
            return Strands(position=position, rotation=rotation, length=length)
        elif strand_repr == 'residual':  # residual
            return Strands(residual=tensor)
        else:
            raise RuntimeError(f'representation {strand_repr} is not supported')

    def to_tensor(self) -> torch.Tensor:
        """ Concatenate strands arrtibutes to a torch Tensor. """
        lst = []
        for f in fields(self):
            attr = getattr(self, f.name)
            if attr is not None:
                lst.append(attr[..., None] if f.name == 'length' else attr)

        return torch.cat(lst, dim=-1)
