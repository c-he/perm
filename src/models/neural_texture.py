import torch
import torch.nn.functional as F

from hair import Strands
from models.encoder import EncodingNetwork
from models.networks_stylegan2 import FullyConnectedLayer
from models.networks_stylegan2 import Generator as StyleGAN2Backbone
from models.networks_stylegan2 import SynthesisNetwork
from models.strand_codec import StrandCodec
from models.unet import UNetModel
from torch_utils import persistence
from utils.blend_shape import sample


@persistence.persistent_class
class RawNeuralTexture(torch.nn.Module):
    def __init__(
        self,
        z_dim,                           # Input latent dimensionality.
        w_dim,                           # Intermediate latent (W) dimensionality.
        img_resolution,                  # Output resolution.
        img_channels,                    # Number of output color channels.
        mapping_kwargs={},               # Arguments for MappingNetwork.
        strand_kwargs={},                # Arguments for StrandCodec.
        **synthesis_kwargs,              # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.backbone = StyleGAN2Backbone(z_dim=z_dim, c_dim=0, w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.decoder = PCADecoder(in_features=img_channels, out_features=img_channels, decoder_lr_mul=1)
        self.strand_codec = StrandCodec(**strand_kwargs)

    def mapping(self, z, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        return self.backbone.mapping(z, c=None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, update_emas=False, **synthesis_kwargs):
        feature_image = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        N = feature_image.shape[0]
        feature_samples = feature_image.permute(0, 2, 3, 1).reshape(N, -1, self.img_channels).contiguous()
        out = self.decoder(feature_samples)
        H = W = self.img_resolution
        coeff_image = out['coeff'].permute(0, 2, 1).reshape(N, -1, H, W).contiguous()
        mask_image = out['mask'].permute(0, 2, 1).reshape(N, 1, H, W).contiguous()
        return {'image': coeff_image, 'image_mask': mask_image}

    def forward(self, z, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Synthesis a batch of neural textures.
        ws = self.mapping(z, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)

    def sample(self, image):
        batch_size = image.shape[0]
        num_coords = image.shape[2] * image.shape[3]
        coeff = image.permute(0, 2, 3, 1).reshape(-1, self.strand_codec.num_coeff)
        position = self.strand_codec.decode(coeff)
        position = position.reshape(batch_size, num_coords, -1, 3)
        position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)

        return Strands(position=position)


@persistence.persistent_class
class ResNeuralTexture(torch.nn.Module):
    def __init__(
        self,
        z_dim,                           # Unused.
        w_dim,                           # Intermediate latent (W) dimensionality.
        img_resolution,                  # Output resolution.
        raw_channels,                    # Number of low-rank coeff channels.
        img_channels,                    # Number of output color channels.
        img_scale,                       # Whether to scale input/output images.
        mapping_kwargs={},               # Argiments for EncodingNetwork.
        strand_kwargs={},                # Arguments for StrandCodec.
        **synthesis_kwargs,              # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim  # unused
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.raw_channels = raw_channels
        self.img_channels = img_channels
        self.img_scale = img_scale
        self.backbone = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.backbone.num_ws
        self.encoder = EncodingNetwork(in_features=img_channels + 1, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.decoder = PCADecoder(in_features=img_channels, out_features=img_channels, decoder_lr_mul=1)
        self.strand_codec = StrandCodec(**strand_kwargs)

    def encode(self, img, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        image = img['image']
        if self.img_scale:
            image = (image - self.mean) / self.std
        mask = img['image_mask'] * 2 - 1  # [0, 1] -> [-1, 1]
        x = torch.cat([image, mask], dim=1)
        return self.encoder(x, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, update_emas=False, **synthesis_kwargs):
        feature_image = self.backbone(ws, update_emas=update_emas, **synthesis_kwargs)
        N = feature_image.shape[0]
        feature_samples = feature_image.permute(0, 2, 3, 1).reshape(N, -1, self.img_channels).contiguous()
        out = self.decoder(feature_samples)
        H = W = self.img_resolution
        coeff_image = out['coeff'].permute(0, 2, 1).reshape(N, -1, H, W).contiguous()
        mask_image = out['mask'].permute(0, 2, 1).reshape(N, 1, H, W).contiguous()
        if self.img_scale:
            coeff_image = coeff_image * self.std + self.mean
        return {'image': coeff_image, 'image_mask': mask_image}

    def forward(self, img, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws, _, _ = self.encode(img, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)

    def sample(self, image, coordinates, mode='nearest'):
        batch_size = image.shape[0]
        num_coords = coordinates.shape[1]
        if coordinates.shape[0] != batch_size:
            coordinates = coordinates.expand(batch_size, -1, -1)
        coeff = sample(coordinates, image, mode=mode)
        position = self.strand_codec.decode(coeff.reshape(batch_size * num_coords, -1))
        position = position.reshape(batch_size, num_coords, -1, 3)
        position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)

        return Strands(position=position)


@persistence.persistent_class
class NeuralTextureSuperRes(torch.nn.Module):
    def __init__(
        self,
        z_dim,                           # Unused.
        w_dim,                           # Unused.
        raw_resolution,                  # Input resolution.
        img_resolution,                  # Output resolution.
        img_channels,                    # Number of output color channels.
        sr_mode,                         # Output mode for super resolution.
        mapping_kwargs={},               # Unused.
        strand_kwargs={},                # Arguments for StrandCodec.
        **synthesis_kwargs,              # Unused.
    ):
        super().__init__()
        self.z_dim = z_dim  # unused
        self.w_dim = w_dim  # unused
        self.raw_resolution = raw_resolution
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.sr_mode = sr_mode
        if sr_mode == 'weight':
            out_channels = 4
        elif sr_mode == 'coeff':
            out_channels = img_channels
        else:
            out_channels = 4 + img_channels
        self.backbone = UNetModel(in_channels=img_channels + 1, out_channels=out_channels, bilinear=True)
        self.strand_codec = StrandCodec(**strand_kwargs)

        # KNN index required for blending.
        u, v = torch.meshgrid(torch.linspace(0, 1, steps=self.img_resolution),
                              torch.linspace(0, 1, steps=self.img_resolution), indexing='ij')
        uv = torch.dstack((u, v)).permute(2, 1, 0)  # (2, H, W)
        uv_guide = F.interpolate(uv.unsqueeze(0), size=(self.raw_resolution, self.raw_resolution), mode='nearest')[0]  # (2, 32, 32)

        uv = uv.permute(1, 2, 0).reshape(-1, 2)  # (H x W, 2)
        uv_guide = uv_guide.permute(1, 2, 0).reshape(-1, 2)
        dist = torch.norm(uv.unsqueeze(1) - uv_guide.unsqueeze(0), dim=-1)  # (H x W, 32 x 32)
        knn_dist, knn_index = dist.topk(4, largest=False)
        self.register_buffer('knn_index', knn_index.flatten())

    def blend(self, image, weight):
        N = image.shape[0]
        guide = image.reshape(N, self.img_channels, -1)
        guide = guide.index_select(dim=-1, index=self.knn_index)
        guide = guide.reshape(N, self.img_channels, self.img_resolution, self.img_resolution, 4)
        return torch.einsum('nchwx,nxhw->nchw', guide, weight)

    def forward(self, img):
        image_raw = img['image_raw']
        mask = img['image_mask'] * 2 - 1  # [0, 1] -> [-1, 1]
        x = torch.cat([image_raw, mask], dim=1)
        upsampled = F.interpolate(x, (self.img_resolution, self.img_resolution), mode='bilinear', align_corners=False)
        superres = self.backbone(upsampled)
        if self.sr_mode == 'weight':
            coeff_image = self.blend(image=image_raw, weight=superres)
            return {'image': coeff_image, 'image_weight': superres}
        elif self.sr_mode == 'coeff':
            return {'image': superres}
        elif self.sr_mode == 'hybrid':
            coeff_image = self.blend(image=image_raw, weight=superres[:, :4]) + superres[:, 4:]
            return {'image': coeff_image, 'image_weight': superres[:, :4], 'image_reg': superres[:, 4:]}

    def sample(self, image, coordinates, mode='nearest'):
        batch_size = image.shape[0]
        num_coords = coordinates.shape[1]
        if coordinates.shape[0] != batch_size:
            coordinates = coordinates.expand(batch_size, -1, -1)
        coeff = sample(coordinates, image, mode=mode)
        position = self.strand_codec.decode(coeff.reshape(batch_size * num_coords, -1))
        position = position.reshape(batch_size, num_coords, -1, 3)
        position = F.pad(position, (0, 0, 1, 0), mode='constant', value=0)

        return Strands(position=position)


@persistence.persistent_class
class PCADecoder(torch.nn.Module):
    def __init__(self, in_features, out_features, **decoder_kwargs):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(in_features, self.hidden_dim, lr_multiplier=decoder_kwargs['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, out_features + 1, lr_multiplier=decoder_kwargs['decoder_lr_mul'])
        )

    def forward(self, sampled_features):
        N, M = sampled_features.shape[:2]
        x = sampled_features
        x = x.view(N * M, -1)
        x = self.net(x)
        x = x.view(N, M, -1)
        coeff = x[..., 1:]
        mask = torch.sigmoid(x[..., :1]) * (1 + 2 * 0.001) - 0.001

        return {'coeff': coeff, 'mask': mask}
