from typing import List, Optional, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import trimesh

from utils.misc import EPSILON
from utils.misc import copy2cpu as c2c


class HairRoots:
    def __init__(self, head_mesh: str, scalp_bounds: Optional[List[float]] = None, scalp_mask: Optional[str] = None) -> None:
        self.head = trimesh.load(head_mesh)
        self.centroid = (self.head.bounds[0] + self.head.bounds[1]) / 2.0
        self.centroid[1] = 0.0  # fix y coordinate to be zero to avoid translation along y axis
        self.centroid = torch.tensor(self.centroid, dtype=torch.float32)
        self.scalp_bounds = scalp_bounds
        if scalp_mask is not None:
            mask = np.array(PIL.Image.open(scalp_mask).convert('L'))
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]  # HW => HWC
            mask = mask.transpose(2, 0, 1)  # HWC => CHW
            self.scalp_mask = torch.tensor(mask / 255.0, dtype=torch.float32)
        else:
            self.scalp_mask = None

    def cartesian_to_spherical(self, x: torch.Tensor) -> torch.Tensor:
        """ Parameterize the scalp surface by considering it as the upper half of a sphere.
        Reference: Wang, Lvdi, et al. "Example-based hair geometry synthesis." ACM SIGGRAPH 2009 papers

        Args:
            x (torch.Tensor): Cartesian points of shape (..., 3).

        Returns:
            (torch.Tensor): Spherical coordinates uvw of shape (..., 3).
        """
        if self.centroid.device != x.device:
            self.centroid = self.centroid.to(x.device)

        x_prime = x - self.centroid
        w = torch.norm(x_prime, dim=-1)
        p = x_prime / (w[..., None] + EPSILON)
        u = torch.acos(p[..., 0] / (p[..., 0] ** 2 + (p[..., 1] + 1) ** 2).sqrt()) / np.pi
        v = torch.acos(p[..., 2] / (p[..., 2] ** 2 + (p[..., 1] + 1) ** 2).sqrt()) / np.pi
        uvw = torch.stack([u, v, w], dim=-1)

        return uvw

    def spherical_to_cartesian(self, x: torch.Tensor) -> torch.Tensor:
        """ Remap spherical coordinates to Cartesian coordinates on the scalp.
        Reference: Wang, Lvdi, et al. "Example-based hair geometry synthesis." ACM SIGGRAPH 2009 papers

        Args:
            x (torch.Tensor): Spherical coordinates of shape (..., 2) or (..., 3).

        Returns:
            (torch.Tensor): Cartesian coordinates xyz of shape (..., 3).
        """
        uv = x[..., :2] * np.pi
        cot_u = 1.0 / torch.tan(uv[..., 0])
        cot_v = 1.0 / torch.tan(uv[..., 1])

        h = 2 / (cot_u ** 2 + cot_v ** 2 + 1)
        p = torch.zeros(*uv.shape[:-1], 3, device=uv.device)  # p is the list of remapped points on the unit sphere
        p[..., 0] = h * cot_u
        p[..., 1] = h - 1
        p[..., 2] = h * cot_v

        if x.shape[-1] == 3:
            if self.centroid.device != x.device:
                self.centroid = self.centroid.to(x.device)
            # the displacement coordinate w is given, just multiply it
            xyz = p * x[..., 2:] + self.centroid
        else:
            with torch.no_grad():
                # shoot rays from the origin to compute their displacements
                extra_dims = p.shape[:-1]
                p = p.reshape(-1, 3)
                o = self.centroid.reshape(1, 3).expand(p.shape[0], -1)  # origins for the rays
                locations, index_ray, index_tri = self.head.ray.intersects_location(ray_origins=c2c(o), ray_directions=c2c(p), multiple_hits=False)
                index_ray = index_ray.argsort()  # sort the indices to make sure the returning locations are in the correct order

                locations = torch.tensor(locations[index_ray], dtype=torch.float32, device=uv.device)
                xyz = locations.reshape(*extra_dims, 3)

        return xyz

    def load_txt(self, fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Load root positions and normals from .txt files.

        Args:
            fname (str): File to load.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Root positions and normals.
        """
        data = np.loadtxt(fname, skiprows=1)
        position = torch.tensor(data[::2], dtype=torch.float32)
        normal = torch.tensor(data[1::2], dtype=torch.float32)

        return position, F.normalize(normal, dim=-1)

    def uv(self, width: int, height: int, include_normal: bool = True) -> np.ndarray:
        """ Compute uv maps for the head mesh.

        Args:
            width (int): Width of the uv map.
            height (int): Height of the uv map.
            include_normal (bool): If true, include normal vectors in the uv map. (default=True)

        Returns:
            (np.ndarray): uv maps of shape (height, width, 3) (storing the xyz coordinates of each uv point) or shape (height, width, 6) (with 3 extra channels for normal vectors).
        """
        assert self.scalp_bounds is not None, "AABB is required but it's not defined yet"
        u, v = np.meshgrid(np.linspace(self.scalp_bounds[0], self.scalp_bounds[1], num=width),
                           np.linspace(self.scalp_bounds[2], self.scalp_bounds[3], num=height), indexing='ij')
        uv = np.dstack((u, v)).reshape(-1, 2)  # (width x height, 2)

        uv *= np.pi
        cot_u = 1.0 / np.tan(uv[:, 0])
        cot_v = 1.0 / np.tan(uv[:, 1])

        h = 2 / (cot_u ** 2 + cot_v ** 2 + 1)
        p = np.zeros((width * height, 3))  # p is the list of remapped points on the unit sphere
        p[:, 0] = h * cot_u
        p[:, 1] = h - 1
        p[:, 2] = h * cot_v

        o = self.centroid.reshape(1, 3).expand(p.shape[0], -1)  # origins for the rays
        locations, index_ray, index_tri = self.head.ray.intersects_location(ray_origins=c2c(o), ray_directions=p, multiple_hits=False)
        index_ray = index_ray.argsort()  # sort the indices to make sure the returning locations are in the correct order
        points = locations[index_ray]  # (width * height, 3)

        if include_normal:
            bary = trimesh.triangles.points_to_barycentric(triangles=self.head.triangles[index_tri], points=points)
            normals = self.head.vertex_normals[self.head.faces[index_tri]]
            normals = trimesh.unitize((normals * bary.reshape((-1, 3, 1))).sum(axis=1))
            texture = np.concatenate([points, normals], axis=-1)
        else:
            texture = points

        return texture.reshape(width, height, -1).transpose((1, 0, 2))

    def surface_normals(self, points: np.ndarray, index_tri: np.ndarray) -> np.ndarray:
        """ Compute normals for points on the mesh surface.

        Args:
            points (np.ndarray): Points on the mesh surface of shape (n, 3).
            index_tri (np.ndarray): Triangle indices associated with points, of shape (n,).

        Returns:
            (np.ndarray): Surface normals of shape (n, 3).
        """
        bary = trimesh.triangles.points_to_barycentric(triangles=self.head.triangles[index_tri], points=points)
        normals = self.head.vertex_normals[self.head.faces[index_tri]]

        return trimesh.unitize((normals * bary.reshape((-1, 3, 1))).sum(axis=1))

    def bounds(self, roots: torch.Tensor) -> None:
        """ Compute AABB of all 2D hair roots in the dataset.

        Args:
            roots (torch.Tensor): Hair roots uv of shape (..., 2).
        """
        u_min = roots[..., 0].min()
        u_max = roots[..., 0].max()
        v_min = roots[..., 1].min()
        v_max = roots[..., 1].max()

        self.scalp_bounds = [u_min, u_max, v_min, v_max]
        print(self.scalp_bounds)

    def rescale(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """ Rescale hair uv coordinates according to the pre-computed AABB.

        Args:
            x (torch.Tensor): uv coordinates of shape (..., 2) or (..., 3).
            inverse (bool): If true, rescale uv coordinates from the scalp bound to the original space (default=False).

        Returns:
            (torch.Tensor): Scaled uv coordinates in the range [0, 1] x [0, 1]
        """
        assert self.scalp_bounds is not None, "AABB is required but it's not defined yet"

        res = x.clone()
        if inverse:
            res[..., 0] = x[..., 0] * (self.scalp_bounds[1] - self.scalp_bounds[0]) + self.scalp_bounds[0]
            res[..., 1] = x[..., 1] * (self.scalp_bounds[3] - self.scalp_bounds[2]) + self.scalp_bounds[2]
        else:
            res[..., 0] = (x[..., 0] - self.scalp_bounds[0]) / (self.scalp_bounds[1] - self.scalp_bounds[0])
            res[..., 1] = (x[..., 1] - self.scalp_bounds[2]) / (self.scalp_bounds[3] - self.scalp_bounds[2])

        res[..., 0] = torch.clamp(res[..., 0], min=0.0, max=1.0)
        res[..., 1] = torch.clamp(res[..., 1], min=0.0, max=1.0)

        return res

    def to(self, device: torch.device) -> None:
        if self.centroid is not None:
            self.centroid = self.centroid.to(device)
        if self.scalp_mask is not None:
            self.scalp_mask = self.scalp_mask.to(device)
