import torch
import torch.nn as nn
from models.networks_stylegan2 import FullyConnectedLayer
from torch_utils import persistence


@persistence.persistent_class
class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return h * x


@persistence.persistent_class
class Bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, out_channels, stride):
        super().__init__()
        if in_channel == out_channels:
            self.shortcut = nn.MaxPool2d(1, stride)
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channels, (1, 1), stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channels, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEModule(out_channels, 16)
        )

    def forward(self, x):
        h = self.shortcut(x)
        x = self.conv(x)
        return x + h


@persistence.persistent_class
class EncodingNetwork(nn.Module):
    def __init__(
        self,
        in_features,         # Input feature channels.
        w_dim,               # Intermediate latent (W) dimensionality.
        num_ws,              # Number of intermediate latents to output, None = do not broadcast.
        num_layers,          # Number of mapping layers.
        space='w',           # Output latent space, either w or w+.
        lr_multiplier=0.01,  # Learning rate multiplier for the mapping layers.
        w_avg_beta=0.998,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.in_features = in_features
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.space = space
        self.w_avg_beta = w_avg_beta

        self.embed = nn.Sequential(
            nn.Conv2d(in_features, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.channels_dict = [min(64 * (2**i), 512) for i in range(num_layers + 1)]
        for idx in range(num_layers):
            in_channels = self.channels_dict[idx]
            out_channels = self.channels_dict[idx + 1]
            layer = Bottleneck_IR_SE(in_channel=in_channels, out_channels=out_channels, stride=2)
            setattr(self, f'conv{idx}', layer)
        self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        if space == 'w':
            self.fc = FullyConnectedLayer(self.channels_dict[-1] * 4 * 4, w_dim * 2, activation='linear', lr_multiplier=lr_multiplier)
        else:
            self.fc = FullyConnectedLayer(self.channels_dict[-1] * 4 * 4, w_dim * num_ws * 2, activation='linear', lr_multiplier=lr_multiplier)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def reparameterize(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * log_sigma)

        return mu + eps * std

    def forward(self, x, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        x = self.embed(x)

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'conv{idx}')
            x = layer(x)
        x = self.pool(x)
        x = x.view(-1, self.channels_dict[-1] * 4 * 4)
        x = self.fc(x)
        if self.space == 'w+':
            x = x.view(-1, self.num_ws, self.w_dim * 2)
        mu, log_sigma = x[..., :self.w_dim], x[..., self.w_dim:]
        w = self.reparameterize(mu, log_sigma)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                if self.space == 'w':
                    self.w_avg.copy_(mu.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))
                else:
                    self.w_avg.copy_(mu.detach().mean(dim=(0, 1)).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None and self.space == 'w':
            with torch.autograd.profiler.record_function('broadcast'):
                w = w.unsqueeze(1).repeat([1, self.num_ws, 1])
                mu = mu.unsqueeze(1).repeat([1, self.num_ws, 1])
                log_sigma = log_sigma.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    w = self.w_avg.lerp(w, truncation_psi)
                else:
                    w[:, :truncation_cutoff] = self.w_avg.lerp(w[:, :truncation_cutoff], truncation_psi)
        return w, mu, log_sigma
