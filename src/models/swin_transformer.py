# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from models.embedding import SinusoidalPositionalEmbedding
from models.networks_stylegan2 import FullyConnectedLayer, MappingNetwork, ToRGBLayer
from torch_utils import misc, persistence
from torch_utils.ops import upfirdn2d


# @persistence.persistent_class
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# @persistence.persistent_class
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q.shape
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


# @persistence.persistent_class
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(in_channel)
        self.affine = FullyConnectedLayer(w_dim, in_channel * 2)

    def forward(self, input, w):
        styles = self.affine(w).unsqueeze(-1)
        gamma, beta = styles.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta
        return out


# @persistence.persistent_class
class StyleSwinTransformerBlock(nn.Module):
    r""" StyleSwin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        w_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, w_dim=512):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2
        self.w_dim = w_dim
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = AdaptiveInstanceNorm(dim, w_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=(self.window_size, self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=(self.window_size, self.window_size), num_heads=num_heads // 2,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])

        attn_mask1 = None
        attn_mask2 = None
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask2 = attn_mask2.masked_fill(
                attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))

        self.register_buffer("attn_mask1", attn_mask1)
        self.register_buffer("attn_mask2", attn_mask2)

        self.norm2 = AdaptiveInstanceNorm(dim, w_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, w):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Double Attn
        shortcut = x
        x = self.norm1(x.transpose(-1, -2), w).transpose(-1, -2)

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)

        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, k1_windows, v1_windows, self.attn_mask1)
        x2 = self.attn[1](q2_windows, k2_windows, v2_windows, self.attn_mask2)

        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x.transpose(-1, -2), w).transpose(-1, -2))

        return x

    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


# @persistence.persistent_class
class StyleBasicLayer(nn.Module):
    """ A basic StyleSwin layer for one stage.

    Args:
        dim (int): Number of input channels.
        img_channels (int): Number of channels for output image.
        resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        out_dim (int): Number of output channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        w_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, img_channels, resolution, depth, num_heads, window_size, out_dim=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., upsample=None,
                 use_checkpoint=False, w_dim=512, resample_filter=[1, 3, 3, 1], fused_modconv_default=True, **layer_kwargs):

        super().__init__()
        self.dim = dim
        self.img_channels = img_channels
        self.resolution = resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        # build blocks
        self.blocks = nn.ModuleList([
            StyleSwinTransformerBlock(dim=dim, input_resolution=resolution,
                                      num_heads=num_heads, window_size=window_size,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop, w_dim=w_dim)
            for _ in range(depth)])
        self.num_block = depth

        self.torgb = ToRGBLayer(dim, img_channels, w_dim)
        self.num_torgb = 1

        if upsample is not None:
            self.upsample = upsample(resolution, dim=dim, out_dim=out_dim)
        else:
            self.upsample = None

    def forward(self, x, img, ws, fused_modconv=None, update_emas=False, **layer_kwargs):
        _ = update_emas  # unused
        w_iter = iter(ws.unbind(dim=1))
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Main layers.
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, next(w_iter))
            else:
                x = block(x, next(w_iter))

        # ToRGB.
        b, n, c = x.shape
        h, w = int(math.sqrt(n)), int(math.sqrt(n))
        y = self.torgb(x.transpose(-1, -2).reshape(b, c, h, w), next(w_iter), fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        if img is not None:
            misc.assert_shape(img, [None, self.img_channels, self.resolution[0] // 2, self.resolution[1] // 2])
            img = upfirdn2d.upsample2d(img, self.resample_filter)
        img = img.add_(y) if img is not None else y

        # Upsample.
        if self.upsample is not None:
            x = self.upsample(x)

        return x, img

    def extra_repr(self) -> str:
        return f"dim={self.dim}, img_channels={self.img_channels}, resolution={self.resolution}, depth={self.depth}"


# @persistence.persistent_class
class BilinearUpsample(nn.Module):
    """ BilinearUpsample Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
    """

    def __init__(self, input_resolution, dim, out_dim=None):
        super().__init__()
        assert dim % 2 == 0, f"x dim are not even."
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.norm = nn.LayerNorm(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sin_pos_embed = SinusoidalPositionalEmbedding(embedding_dim=out_dim // 2, padding_idx=0, init_size=out_dim // 2)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert C == self.dim, "wrong in PatchMerging"

        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()   # B,C,H,W
        x = self.upsample(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L * 4, C)   # B,H,W,C
        x = self.norm(x)
        x = self.reduction(x)

        # Add SPE
        x = x.reshape(B, H * 2, W * 2, self.out_dim).permute(0, 3, 1, 2)
        x += self.sin_pos_embed.make_grid2d(H * 2, W * 2, B) * self.alpha
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * 2 * W * 2, self.out_dim)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


# @persistence.persistent_class
class StyleSwinTransformer(nn.Module):
    def __init__(
        self,
        w_dim,
        img_resolution,
        img_channels,
        channel_multiplier=2,
        enable_full_resolution=8,
        mlp_ratio=4,
        use_checkpoint=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        fused_modconv_default=True,
        raw_resolution=None,
        **layer_kwargs
    ):
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.mlp_ratio = mlp_ratio

        start_idx = 2 if raw_resolution is None else int(np.log2(raw_resolution))
        end_idx = int(np.log2(img_resolution))
        full_res_idx = int(np.log2(enable_full_resolution))
        window_sizes = [2 ** i if i <= full_res_idx else 8 for i in range(start_idx, end_idx + 1)]

        block_depths = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        block_channels = [
            # 512,
            # 512,
            # 512,
            # 512,
            256 * channel_multiplier,
            128 * channel_multiplier,
            64 * channel_multiplier,
            32 * channel_multiplier,
            16 * channel_multiplier
        ]
        num_heads = [max(c // 32, 4) for c in block_channels]

        self.const = torch.nn.Parameter(torch.randn([block_channels[0], 4, 4]))
        self.layers = nn.ModuleList()

        self.num_ws = 0
        for i_layer in range(start_idx, end_idx + 1):
            if i_layer == start_idx:
                in_channels = block_channels[0] if start_idx == 0 else img_channels
            else:
                in_channels = block_channels[i_layer - start_idx - 1]
            out_channels = block_channels[i_layer - start_idx]

            layer = StyleBasicLayer(dim=in_channels,
                                    img_channels=img_channels,
                                    resolution=(2 ** i_layer, 2 ** i_layer),
                                    depth=block_depths[i_layer - start_idx],
                                    num_heads=num_heads[i_layer - start_idx],
                                    window_size=window_sizes[i_layer - start_idx],
                                    out_dim=out_channels,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                    upsample=BilinearUpsample if (i_layer < end_idx) else None,
                                    use_checkpoint=use_checkpoint,
                                    w_dim=w_dim,
                                    fused_modconv_default=fused_modconv_default, **layer_kwargs)
            self.layers.append(layer)
            self.num_ws += layer.num_block + layer.num_torgb

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, img, ws, **layer_kwargs):
        layer_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for layer in self.layers:
                layer_ws.append(ws.narrow(1, w_idx, layer.num_block + layer.num_torgb))
                w_idx += layer.num_block + layer.num_torgb

        if x is None:
            x = self.const.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        for layer, cur_ws in zip(self.layers, layer_ws):
            x, img = layer(x, img, cur_ws, **layer_kwargs)

        return img


# @persistence.persistent_class
class Upsampler(nn.Module):
    def __init__(
        self,
        z_dim,                      # Input latent (Z) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        raw_resolution,             # Input resolution.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs={},          # Arguments for MappingNetwork.
        **transformer_kwargs,       # Arguments for StyleSwinTransformer.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.raw_resolution = raw_resolution
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.superresolution = StyleSwinTransformer(w_dim=w_dim, raw_resolution=raw_resolution, img_resolution=img_resolution, img_channels=img_channels, **transformer_kwargs)
        self.num_ws = self.superresolution.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=0, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, x, z, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c=None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.superresolution(x, img=None, ws=ws, update_emas=update_emas, **synthesis_kwargs)
        return img
