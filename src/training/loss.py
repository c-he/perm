# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.discriminator import filtered_resizing
from hair.loss import StrandGeometricLoss
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix, upfirdn2d

# ----------------------------------------------------------------------------


class Loss:
    def accumulate_gradients(self, **kwargs):  # to be overridden by subclass
        raise NotImplementedError()

# ----------------------------------------------------------------------------


class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, r1_gamma_mask=10, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.r1_gamma_mask = r1_gamma_mask

        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.blur_raw_target = True

    def run_G(self, z, update_emas=False):
        ws = self.G.mapping(z, update_emas=update_emas)
        return self.G.synthesis(ws, update_emas=update_emas)

    def run_D(self, img, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            img['image'] = self.augment_pipe(img['image'])

        logits = self.D(img, c=None, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_mask, gen_z, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        real_img = {'image': real_img, 'image_mask': real_mask}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z)
                gen_logits = self.run_D(gen_img, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, update_emas=True)
                gen_logits = self.run_D(gen_img, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_mask = real_img['image_mask'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_mask': real_img_tmp_image_mask}

                real_logits = self.run_D(real_img_tmp, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_mask']],
                                                       create_graph=True, only_inputs=True)
                        r1_grads_image = r1_grads[0]
                        r1_grads_image_mask = r1_grads[1]
                    r1_penalty = r1_grads_image.square().sum([1, 2, 3])
                    r1_penalty_mask = r1_grads_image_mask.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2) + r1_penalty_mask * (self.r1_gamma_mask / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/r1_penalty_mask', r1_penalty_mask)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------


class VAELoss(Loss):
    def __init__(self, device, G, D, augment_pipe, r1_gamma=10, blur_init_sigma=0, blur_fade_kimg=0, lambda_tex=1, lambda_geo=0, lambda_kl=0):
        super().__init__()
        self.device = device
        self.G = G

        self.lambda_tex = lambda_tex
        self.lambda_geo = lambda_geo
        self.lambda_kl = lambda_kl

        self.strand_loss = StrandGeometricLoss()

    def run_G(self, img, update_emas=False):
        ws, mu, log_sigma = self.G.encode(img, update_emas=update_emas)
        out = self.G.synthesis(ws, noise_mode='const', update_emas=update_emas)
        out.update(mu=mu, log_sigma=log_sigma)
        return out

    def strand_geo_loss(self, gen_img, real_img, coords):
        strands = self.G.sample(gen_img, coords, mode='nearest')
        strands_gt = self.G.sample(real_img, coords, mode='nearest')
        return self.strand_loss(strands, strands_gt)

    def kl_loss(self, mu, log_sigma):
        loss = 0.5 * torch.sum(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma, dim=-1)
        return loss.mean()

    def accumulate_gradients(self, phase, real_img, real_mask, coords, gain, cur_nimg):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            real_img_raw = real_img[:, :self.G.raw_channels]
            real_img_res = real_img[:, self.G.raw_channels:]
            real_img = {'image': real_img_res, 'image_mask': real_mask}
            gen_img = self.run_G(real_img, update_emas=True)

            tex_recon = torch.nn.functional.l1_loss(gen_img['image'], real_img['image']) + torch.nn.functional.l1_loss(gen_img['image_mask'], real_img['image_mask'])
            training_stats.report('Loss/G/tex', tex_recon)
            geo_recon = 0
            if self.lambda_geo > 0:
                real_image = torch.cat([real_img_raw, real_img['image']], dim=1)
                gen_image = torch.cat([real_img_raw, gen_img['image']], dim=1)
                terms = self.strand_geo_loss(gen_image, real_image, coords)
                for k, v in terms.items():
                    geo_recon += v
                    training_stats.report(f'Loss/G/{k}', v)
            loss_kl = 0
            if self.lambda_kl > 0:
                loss_kl = self.kl_loss(gen_img['mu'], gen_img['log_sigma'])
                training_stats.report(f'Loss/G/KL', loss_kl)

            loss_Gmain = self.lambda_tex * tex_recon + self.lambda_geo * geo_recon + self.lambda_kl * loss_kl
            training_stats.report('Loss/G/loss', loss_Gmain)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss_Gmain.mean().mul(gain).backward()

# ----------------------------------------------------------------------------


class UNetSuperResLoss(Loss):
    def __init__(self, device, G, D, augment_pipe, r1_gamma=10, blur_init_sigma=0, blur_fade_kimg=0, lambda_tex=1, lambda_geo=0, lambda_reg=0):
        super().__init__()
        self.device = device
        self.G = G

        self.lambda_tex = lambda_tex
        self.lambda_geo = lambda_geo
        self.lambda_reg = lambda_reg

        self.strand_loss = StrandGeometricLoss()

    def run_G(self, img, update_emas=False):
        return self.G(img)

    def strand_geo_loss(self, gen_img, real_img, coords):
        strands = self.G.sample(gen_img, coords, mode='nearest')
        strands_gt = self.G.sample(real_img, coords, mode='nearest')
        return self.strand_loss(strands, strands_gt)

    def accumulate_gradients(self, phase, real_img, real_img_raw, real_mask, real_mask_raw, coords, gain, cur_nimg):
        with torch.autograd.profiler.record_function('Gmain_forward'):
            real_image = {'image_raw': real_img_raw, 'image_mask': real_mask_raw}
            gen_img = self.run_G(real_image, update_emas=True)

            tex_recon = torch.nn.functional.l1_loss(gen_img['image'], real_img[:, :self.G.img_channels])
            training_stats.report('Loss/G/tex', tex_recon)
            geo_recon = 0
            if self.lambda_geo > 0:
                terms = self.strand_geo_loss(gen_img['image'], real_img[:, :self.G.img_channels], coords)
                for k, v in terms.items():
                    geo_recon += v
                    training_stats.report(f'Loss/G/{k}', v)
            loss_reg = 0
            if self.lambda_reg > 0 and 'image_reg' in gen_img:
                loss_reg = (gen_img['image_reg'] ** 2).mean()
                training_stats.report(f'Loss/G/reg', loss_reg)

            loss_Gmain = self.lambda_tex * tex_recon + self.lambda_geo * geo_recon + self.lambda_reg * loss_reg
            training_stats.report('Loss/G/loss', loss_Gmain)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            loss_Gmain.mean().mul(gain).backward()

# ----------------------------------------------------------------------------
