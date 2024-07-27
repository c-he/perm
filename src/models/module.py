from typing import List, Optional

import admin_torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embedding import FourierFeatMapping, IdentityMapping, PositionalEncoding
from models.init import *


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


# Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
# special first-layer initialization scheme
NONLINEARITY_AND_INIT = {
    'sine': (Sine(), sine_init, first_layer_sine_init),
    'relu': (nn.ReLU(inplace=True), init_weights_relu, None),
    'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
    'tanh': (nn.Tanh(), init_weights_xavier, None),
    'selu': (nn.SELU(inplace=True), init_weights_selu, None),
    'softplus': (nn.Softplus(), init_weights_normal, None),
    'elu': (nn.ELU(inplace=True), init_weights_elu, None),
    'identity': (nn.Identity(), None, None)
}

""" MLP Modules
"""


class FCBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_norm: bool = True,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.residual = (in_features == out_features)  # when inputs and outputs have the same dimension, build a residual block
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensors of shape (..., in_features).

        Returns:
            (torch.Tensor): Tensors after linear transformation ($y = xA^T + b$) of shape (..., out_features).
        """
        out = self.linear(x)

        if self.layer_norm:
            out = self.layer_norm(out)

        if self.residual:
            out = self.nonlinearity(out + x)
        else:
            out = self.nonlinearity(out)

        return out


class FCBlockMoE(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 1,
        layer_norm: bool = True,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(num_experts, 1, out_features))

        self.residual = (in_features == out_features)  # when inputs and outputs have the same dimension, build a residual block
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensors of shape (batch_size, in_features) or (num_experts, batch_size, in_features).

        Returns:
            (torch.Tensor): Tensors after linear transformation ($y = xA^T + b$) of shape (num_experts, batch_size, out_features).
        """
        out = torch.matmul(x, self.weight.mT) + self.bias  # (num_experts, batch_size, out_features)

        if self.layer_norm:
            out = self.layer_norm(out)

        if self.residual:
            if x.ndim != out.ndim:
                identity = x.unsqueeze(0).expand(out.shape[0], -1, -1)
            else:
                identity = x
            out = self.nonlinearity(out + identity)
        else:
            out = self.nonlinearity(out)

        return out


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        layer_norm: bool = True,
        skip_connection: bool = True,
        outermost_linear: bool = False,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__()

        nl, nl_weight_init, first_layer_init = NONLINEARITY_AND_INIT[nonlinearity]
        if weight_init is None:
            weight_init = nl_weight_init

        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.layers.append(FCBlock(in_features, hidden_features, layer_norm, nonlinearity=nonlinearity))
        for _ in range(hidden_layers):
            self.layers.append(FCBlock(in_features + hidden_features if skip_connection else hidden_features, hidden_features, layer_norm, nonlinearity=nonlinearity))
        if outermost_linear:
            self.layers.append(FCBlock(hidden_features, out_features, layer_norm=False, nonlinearity='identity'))
        else:
            self.layers.append(FCBlock(hidden_features, out_features, layer_norm=False, nonlinearity=nonlinearity))

        self.layers.apply(weight_init)
        if first_layer_init is not None:  # apply special initialization to first layer, if applicable
            self.layers[0].apply(first_layer_init)

    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1, len(self.layers) - 1):
            if self.skip_connection:
                out = self.layers[i](torch.cat((out, x), dim=-1))
            else:
                out = self.layers[i](out)
        out = self.layers[-1](out)

        return out


class MLPMoE(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        num_experts: int = 1,
        layer_norm: bool = True,
        skip_connection: bool = True,
        outermost_linear: bool = False,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__()

        nl, nl_weight_init, first_layer_init = NONLINEARITY_AND_INIT[nonlinearity]
        if weight_init is None:
            weight_init = nl_weight_init

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList()
        self.layers.append(FCBlockMoE(in_features, hidden_features, num_experts, layer_norm, nonlinearity=nonlinearity))
        for _ in range(hidden_layers):
            self.layers.append(FCBlockMoE(in_features + hidden_features if skip_connection else hidden_features, hidden_features, num_experts, layer_norm, nonlinearity=nonlinearity))
        if outermost_linear:
            self.layers.append(FCBlockMoE(hidden_features, out_features, num_experts, layer_norm=False, nonlinearity='identity'))
        else:
            self.layers.append(FCBlockMoE(hidden_features, out_features, num_experts, layer_norm=False, nonlinearity=nonlinearity))

        self.layers.apply(weight_init)
        if first_layer_init is not None:  # apply special initialization to first layer, if applicable
            self.layers[0].apply(first_layer_init)

    def forward(self, x):
        extra_dims = x.shape[:-1]
        identity = x.reshape(1, -1, self.in_features).expand(self.num_experts, -1, -1)

        out = self.layers[0](x.reshape(-1, self.in_features))
        for i in range(1, len(self.layers) - 1):
            if self.skip_connection:
                out = self.layers[i](torch.cat((out, identity), dim=-1))
            else:
                out = self.layers[i](out)
        out = self.layers[-1](out)
        out = out.reshape(self.num_experts, *extra_dims, self.out_features)

        return out


""" INR Modules
"""


class Codebook(nn.Module):
    def __init__(self, num_entries: int, entry_size: int):
        super().__init__()
        self.codebook = nn.Embedding(num_entries, entry_size)

    def forward(self, idx):
        return self.codebook(idx)

    @property
    def data(self):
        return self.codebook.weight


class INRMoE(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        num_experts: int,
        layer_norm: bool = True,
        nonlinearity: str = 'relu',
        pos_embed: str = 'identity',
        num_freqs: int = 7,
        freq_scale: float = 1.
    ):
        super().__init__()

        if pos_embed == 'ffm':
            self.pos_embed = FourierFeatMapping(in_features, num_freqs, freq_scale)
        elif pos_embed == 'pe':
            self.pos_embed = PositionalEncoding(in_features, num_freqs)
        else:
            self.pos_embed = IdentityMapping(in_features)
        in_features = self.pos_embed.out_dim

        self.experts = MLPMoE(in_features, out_features, hidden_layers, hidden_features, num_experts, layer_norm, skip_connection=True, outermost_linear=True, nonlinearity=nonlinearity)

    def forward(self, coords, coeff):
        coords = self.pos_embed(coords)  # (batch_size, num_coords, in_features)
        basis = self.experts(coords)  # (num_experts, batch_size, num_coords, out_features)
        if self.experts.num_experts == 1:
            out = basis.squeeze(0)
        else:
            # coeff = F.softmax(coeff, dim=-1).T  # (num_experts, batch_size)
            coeff = F.normalize(coeff.abs(), dim=-1, p=1).T  # (num_experts, batch_size)
            out = torch.sum(coeff[..., None, None] * basis, dim=0)  # (batch_size, num_coords, out_features)

        return out


class SynthesisMLP(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        layer_norm: bool = True,
        skip_connection: bool = True,
        outermost_linear: bool = False,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__(in_features, out_features, hidden_layers, hidden_features, layer_norm, skip_connection, outermost_linear, nonlinearity, weight_init)

    def forward(self, x, gating_layers=None):
        out = self.layers[0](x)
        for i in range(1, len(self.layers) - 1):
            if self.skip_connection:
                out = self.layers[i](torch.cat((out, x), dim=-1))
            else:
                out = self.layers[i](out)
            if gating_layers:
                out = gating_layers[i - 1] * out
        out = self.layers[-1](out)

        return out


class ModulatorMLP(MLP):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        layer_norm: bool = True,
        skip_connection: bool = True,
        outermost_linear: bool = False,
        nonlinearity: str = 'relu',
        weight_init=None
    ):
        super().__init__(in_features, out_features, hidden_layers, hidden_features, layer_norm, skip_connection, outermost_linear, nonlinearity, weight_init)

    def forward(self, x):
        out = self.layers[0](x)
        gating_layers = []
        for i in range(1, len(self.layers) - 1):
            if self.skip_connection:
                out = self.layers[i](torch.cat((out, x), dim=-1))
            else:
                out = self.layers[i](out)
            gating_layers.append(out)

        return gating_layers


class ModSIREN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: int,
        hidden_features: int,
        latent_dim: int,
        concat: bool = True,
        synthesis_layer_norm: bool = True,
        modulator_layer_norm: bool = True,
        synthesis_nonlinearity: Optional[str] = 'relu',
        modulator_nonlinearity: Optional[str] = 'relu',
        pos_embed: str = 'identity',
        num_freqs: int = 7,
        freq_scale: float = 1.
    ):
        super().__init__()

        if pos_embed == 'ffm':
            self.pos_embed = FourierFeatMapping(in_features, num_freqs, freq_scale)
        elif pos_embed == 'pe':
            self.pos_embed = PositionalEncoding(in_features, num_freqs)
        else:
            self.pos_embed = IdentityMapping(in_features)
        in_features = self.pos_embed.out_dim

        self.concat = concat

        if modulator_nonlinearity:
            self.synthesis = SynthesisMLP(in_features, out_features, hidden_layers, hidden_features, synthesis_layer_norm,
                                          skip_connection=True, outermost_linear=True, nonlinearity=synthesis_nonlinearity)
            if concat:
                self.modulator = ModulatorMLP(latent_dim + in_features, out_features, hidden_layers, hidden_features, modulator_layer_norm,
                                              skip_connection=True, outermost_linear=True, nonlinearity=modulator_nonlinearity)
            else:
                self.modulator = ModulatorMLP(latent_dim, out_features, hidden_layers, hidden_features, modulator_layer_norm,
                                              skip_connection=True, outermost_linear=True, nonlinearity=modulator_nonlinearity)
        else:
            self.synthesis = SynthesisMLP(latent_dim + in_features, out_features, hidden_layers, hidden_features, synthesis_layer_norm,
                                          skip_connection=True, outermost_linear=True, nonlinearity=synthesis_nonlinearity)
            self.modulator = None

    def forward(self, coords, embedding=None):
        coords = self.pos_embed(coords)

        if embedding is not None:
            if embedding.ndim != coords.ndim:
                embedding = embedding.unsqueeze(1).expand(-1, coords.shape[1], -1)

            gating_layers = None
            if self.modulator:
                if self.concat:
                    gating_layers = self.modulator(torch.cat([embedding, coords], dim=-1))
                else:
                    gating_layers = self.modulator(embedding)
                out = self.synthesis(coords, gating_layers)
            else:
                out = self.synthesis(torch.cat([embedding, coords], dim=-1), gating_layers=None)
        else:
            out = self.synthesis(coords)

        return out


""" CNN Modules
"""


class Conv2dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        nonlinearity: str
    ):
        super().__init__()

        kernel_size = kernel_size - 1 if kernel_size % 2 == 0 else kernel_size
        padding = (kernel_size - 1) // 2
        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
            self.nonlinearity,
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
            self.nonlinearity
        )

    def forward(self, x):
        identity = x
        out = self.convs(x)
        out = self.nonlinearity(out + identity)
        return out


class TextureEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        kernel_size: int,
        pooling: str,
        spatial: bool = True,
        batch_norm: bool = False,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        stride = 2
        padding = (kernel_size - 1) // 2
        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]

        self.conv_layers = nn.ModuleList()
        num_layers = len(hidden_features)
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = hidden_features[0]
            else:
                in_channels = hidden_features[i - 1]
                out_channels = hidden_features[i]
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                self.nonlinearity,
                Conv2dResBlock(out_channels, out_channels, kernel_size, nonlinearity)
            ))
        if pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        elif pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pooling = None

        if spatial:
            self.mean = nn.Conv2d(hidden_features[-1], out_features, 1)
            self.logvar = nn.Conv2d(hidden_features[-1], out_features, 1)
        else:
            self.mean = nn.Linear(hidden_features[-1], out_features)
            self.logvar = nn.Linear(hidden_features[-1], out_features)

    def forward(self, x):
        out = x
        for layer in self.conv_layers:
            out = layer(out)

        if self.pooling is not None:
            out = self.pooling(out).flatten(1)

        return self.mean(out), self.logvar(out)


class TextureDecoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        kernel_size: int,
        spatial: bool = True,
        bilinear: bool = True,
        batch_norm: bool = False,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        stride = 2
        padding = (kernel_size - 1) // 2
        output_padding = 0 if kernel_size % 2 == 0 else 1
        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]

        self.spatial = spatial
        if spatial:
            in_features = in_features
        else:
            in_features = in_features // 16

        self.conv_layers = nn.ModuleList()
        num_layers = len(hidden_features)
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = hidden_features[0]
            else:
                in_channels = hidden_features[i - 1]
                out_channels = hidden_features[i]

            if bilinear:
                self.conv_layers.append(nn.Sequential(
                    Conv2dResBlock(in_channels, in_channels, kernel_size, nonlinearity),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(in_channels, out_channels, kernel_size + output_padding - 1, 1, padding),
                    nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                    self.nonlinearity
                ))
            else:
                self.conv_layers.append(nn.Sequential(
                    Conv2dResBlock(in_channels, in_channels, kernel_size, nonlinearity),
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                    nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                    self.nonlinearity
                ))
        self.conv_layers.append(nn.Conv2d(hidden_features[-1], out_features, 1))

    def forward(self, x):
        if self.spatial:
            out = x
        else:
            out = x.reshape(x.shape[0], -1, 4, 4)

        for layer in self.conv_layers:
            out = layer(out)

        return out


""" Transformer Modules
"""


class CrossAttention(nn.Module):
    r"""
    A cross attention layer adapted from: https://github.com/huggingface/diffusers/blob/4125756e88e82370c197fecf28e9f0b4d7eee6c3/src/diffusers/models/cross_attention.py.
    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        offset_attention: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.offset_attention = offset_attention
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Linear(query_dim, query_dim) if self.offset_attention else nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        if self.offset_attention:
            alpha = 1
        else:
            alpha = self.scale

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=alpha
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        if self.offset_attention:
            attention_probs = attention_scores.softmax(dim=1)
            attention_probs = attention_probs / (1e-12 + attention_probs.sum(dim=-1, keepdims=True))
        else:
            attention_probs = attention_scores.softmax(dim=-1)

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length):
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, device=attention_mask.device)
                attention_mask = torch.concat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask

    def forward(self, init_query, embedding, attention_mask=None):
        batch_size, sequence_length, _ = init_query.shape
        attention_mask = self.prepare_attention_mask(attention_mask, sequence_length)

        query = self.to_q(init_query)
        key = self.to_k(embedding)
        value = self.to_v(embedding)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(query, key, attention_mask)
        scores = torch.bmm(attention_probs, value)
        scores = self.batch_to_head_dim(scores)

        if self.offset_attention:
            scores = init_query - self.to_out[0](scores)

        # linear proj
        scores = self.to_out[-2](scores)
        # dropout
        scores = self.to_out[-1](scores)

        return scores


class LayeredAttention(nn.Module):
    def __init__(
            self,
            query_dim: int,
            feature_dim: int,
            num_heads: int,
            dim_head: int,
            num_layers: int,
            offset_attention: bool = False,
            pre_norm: bool = False,
            admin_init: bool = False,
    ):
        super().__init__()
        self.attention_layers = nn.ModuleList([CrossAttention(query_dim, feature_dim, num_heads, dim_head, offset_attention=offset_attention) for _ in range(num_layers)])
        self.pre_norm = pre_norm
        self.layer_norm = nn.LayerNorm(query_dim)
        self.admin_init = admin_init
        self.residual = admin_torch.as_module(num_layers * 2)

    def forward(self, query, feature):
        for attention_layer in self.attention_layers:
            if self.pre_norm:
                query_to_attention = self.layer_norm(query)
            else:
                query_to_attention = query

            layer_out = attention_layer(query_to_attention, feature)

            if self.admin_init:
                query = self.residual(query, layer_out)
            else:
                query = query + layer_out

            if not self.pre_norm:
                query = self.layer_norm(query)

        return query


class Transformer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_attention_layers: int,
        token_dim: int,
        num_heads: int,
        dim_head: int,
        self_attention: bool,
        offset_attention: bool,
        pre_norm: bool,
        admin_init: bool,
        pos_embed: str = 'identity',
        num_freqs: int = 7,
        freq_scale: float = 1.
    ):
        super().__init__()

        if pos_embed == 'ffm':
            self.pos_embed = FourierFeatMapping(in_features, num_freqs, freq_scale)
        elif pos_embed == 'pe':
            self.pos_embed = PositionalEncoding(in_features, num_freqs)
        else:
            self.pos_embed = IdentityMapping(in_features)
        query_dim = self.pos_embed.out_dim

        self.self_attention = self_attention
        self.token_dim = token_dim
        if self_attention:
            self.attention_layer = LayeredAttention(query_dim + token_dim, query_dim + token_dim, num_heads, dim_head, num_attention_layers, offset_attention, pre_norm, admin_init)
            self.out_layer = MLP(query_dim + token_dim, out_features, hidden_layers=1, hidden_features=256, layer_norm=True, skip_connection=True, outermost_linear=True)
        else:
            self.attention_layer = LayeredAttention(query_dim, token_dim, num_heads, dim_head, num_attention_layers, offset_attention, pre_norm, admin_init)
            self.out_layer = MLP(query_dim, out_features, hidden_layers=1, hidden_features=256, layer_norm=True, skip_connection=True, outermost_linear=True)

    def forward(self, coords, embedding):
        coords = self.pos_embed(coords)
        latent_vec = embedding
        latent_vec = latent_vec.reshape(latent_vec.shape[0], -1, self.token_dim)

        if self.self_attention:
            assert latent_vec.shape[1] == 1
            query = torch.cat((coords, latent_vec.repeat(1, coords.shape[1], 1)), dim=-1)
            attention = self.attention_layer(query, query)
        else:
            attention = self.attention_layer(coords, latent_vec)
        model_output = self.out_layer(attention)

        return model_output


""" Strand Codec Modules
"""


class Conv1dResBlock(nn.Module):
    '''Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py'''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        nonlinearity: str
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, 1, padding),
            self.nonlinearity,
            nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding),
            self.nonlinearity
        )

    def forward(self, x):
        identity = x
        out = self.convs(x)
        out = self.nonlinearity(out + identity)
        return out


class StrandEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        kernel_size: int,
        nonlinearity: str = 'relu',
        pooling: str = 'max'
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]

        self.conv_layers = nn.ModuleList()
        num_layers = len(hidden_features)
        padding = (kernel_size - 1) // 2
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = hidden_features[i]
            else:
                in_channels = hidden_features[i - 1]
                out_channels = hidden_features[i]
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                self.nonlinearity,
                Conv1dResBlock(out_channels, out_channels, kernel_size, nonlinearity)
            ))
        self.conv_layers.append(nn.Conv1d(hidden_features[-1], out_features, 1))
        self.pooling = pooling

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.conv_layers:
            x = layer(x)
        if self.pooling == 'max':
            out = torch.max(x, dim=-1)[0]
        else:
            out = torch.mean(x, dim=-1)

        return out


class StrandDecoder(nn.Module):
    def __init__(
        self,
        seq_length: int,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        kernel_size: int,
        nonlinearity: str = 'relu'
    ):
        super().__init__()

        self.nonlinearity = NONLINEARITY_AND_INIT[nonlinearity][0]

        self.fc = nn.Sequential(
            nn.Linear(1, seq_length),
            self.nonlinearity
        )

        self.conv_layers = nn.ModuleList()
        num_layers = len(hidden_features)
        padding = (kernel_size - 1) // 2
        for i in range(num_layers):
            if i == 0:
                in_channels = in_features
                out_channels = hidden_features[i]
            else:
                in_channels = hidden_features[i - 1]
                out_channels = hidden_features[i]
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                self.nonlinearity,
                Conv1dResBlock(out_channels, out_channels, kernel_size, nonlinearity)
            ))
        self.conv_layers.append(nn.Conv1d(hidden_features[-1], out_features, 1))

    def forward(self, x):
        out = self.fc(x.unsqueeze(-1))
        for layer in self.conv_layers:
            out = layer(out)

        return out.permute(0, 2, 1)
