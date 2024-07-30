# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import copy
import json
import os
import pickle
import time

import numpy as np
import PIL.Image
import torch

import dnnlib
import legacy
from torch_utils import misc, training_stats
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix

# ----------------------------------------------------------------------------


def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(3840 // training_set.img_shape[2], 7, 32)
    gh = np.clip(2160 // training_set.img_shape[1], 4, 32)

    # Show random subset of training samples.
    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    images, images_raw, masks, masks_raw = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(images_raw), np.stack(masks), np.stack(masks_raw)

# ----------------------------------------------------------------------------


def save_image_grid(imgs, fname, grid_size, drange=None):
    images = []
    for img in imgs:
        if drange is None:
            lo, hi = img.min(), img.max()
        else:
            lo, hi = drange
        img = (img - lo) / (hi - lo)
        images.append(img[:3])
    images = np.stack(images)
    images = np.rint(images * 255).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = images.shape
    images = images.reshape([gh, gw, C, H, W])
    images = images.transpose(0, 3, 1, 4, 2)
    images = images.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(images[..., 0], 'L').save(fname)
    else:
        PIL.Image.fromarray(images, 'RGB').save(fname)


# ----------------------------------------------------------------------------


def training_loop(
    run_dir='.',                 # Output directory.
    exp_cfg='raw-tex',           # Experiment configuration.
    training_set_kwargs={},      # Options for training set.
    data_loader_kwargs={},       # Options for torch.utils.data.DataLoader.
    G_kwargs={},                 # Options for generator network.
    D_kwargs={},                 # Options for discriminator network.
    G_opt_kwargs={},             # Options for generator optimizer.
    D_opt_kwargs={},             # Options for discriminator optimizer.
    hair_roots_kwargs={},        # Options for hair roots.
    roots=None,                  # Hair root file.
    augment_kwargs=None,         # Options for augmentation pipeline. None = disable.
    loss_kwargs={},              # Options for loss function.
    metrics=[],                  # Metrics to evaluate during training.
    random_seed=0,               # Global random seed.
    num_gpus=1,                  # Number of GPUs participating in the training.
    rank=0,                      # Rank of the current process in [0, num_gpus[.
    batch_size=4,                # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu=4,                 # Number of samples processed at a time by one GPU.
    ema_kimg=10,                 # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup=0.05,             # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval=None,         # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval=16,           # How often to perform regularization for D? None = disable lazy regularization.
    augment_p=0,                 # Initial value of augmentation probability.
    ada_target=None,             # ADA target value. None = fixed p.
    ada_interval=4,              # How often to perform ADA adjustment?
    ada_kimg=500,                # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg=25000,            # Total length of the training, measured in thousands of real images.
    kimg_per_tick=4,             # Progress snapshot interval.
    image_snapshot_ticks=50,     # How often to save image snapshots? None = disable.
    network_snapshot_ticks=50,   # How often to save network snapshots? None = disable.
    resume_pkl=None,             # Network pickle to resume training from.
    resume_kimg=0,               # First kimg to report when resuming training.
    cudnn_benchmark=True,        # Enable torch.backends.cudnn.benchmark?
    abort_fn=None,               # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn=None,            # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                 # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Number of images:', len(training_set))
        print('Image shape:     ', training_set.img_shape)
        print('Raw image shape: ', training_set.raw_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    if exp_cfg == 'raw-tex':
        G = dnnlib.util.construct_class_by_name(**G_kwargs,
                                                img_channels=training_set.raw_channels, img_resolution=training_set.raw_resolution).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        D = dnnlib.util.construct_class_by_name(**D_kwargs,
                                                img_channels=training_set.raw_channels, img_resolution=training_set.raw_resolution).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        G_ema = copy.deepcopy(G).eval()
    elif exp_cfg == 'res-tex':
        G = dnnlib.util.construct_class_by_name(**G_kwargs, raw_channels=training_set.raw_channels,
                                                img_channels=training_set.img_channels - training_set.raw_channels, img_resolution=training_set.img_resolution).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        coeff_stats = np.load(os.path.join(training_set.path, 'coeff-stats.npz'))
        G.register_buffer('mean', torch.tensor(np.expand_dims(coeff_stats['mean'][G.raw_channels:], axis=(0, 2, 3)), dtype=torch.float32, device=device))
        G.register_buffer('std', torch.tensor(np.expand_dims(coeff_stats['std'][G.raw_channels:], axis=(0, 2, 3)), dtype=torch.float32, device=device))
        D = None
        G_ema = None
    elif exp_cfg == 'super-res':
        G = dnnlib.util.construct_class_by_name(**G_kwargs, img_channels=training_set.raw_channels,
                                                raw_resolution=training_set.raw_resolution, img_resolution=training_set.img_resolution).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        D = None
        G_ema = None

    # Load hair roots.
    if roots is not None:
        hair_roots = dnnlib.util.construct_class_by_name(**hair_roots_kwargs)
        roots, _ = hair_roots.load_txt(roots)
        roots = hair_roots.cartesian_to_spherical(roots)[..., :2]
        coords = hair_roots.rescale(roots).to(device)
        coords = coords.unsqueeze(0)
    else:
        coords = None

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            if module is not None:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        if exp_cfg == 'raw-tex':
            z = torch.empty([batch_gpu, G.z_dim], device=device)
            img = misc.print_module_summary(G, [z])
            misc.print_module_summary(D, [img, None])
        elif exp_cfg == 'res-tex':
            image = torch.torch.empty([batch_gpu, G.img_channels, G.img_resolution, G.img_resolution], device=device)
            mask = torch.torch.empty([batch_gpu, 1, G.img_resolution, G.img_resolution], device=device)
            misc.print_module_summary(G, [{'image': image, 'image_mask': mask}])
        elif exp_cfg == 'super-res':
            raw_image = torch.torch.empty([batch_gpu, G.img_channels, G.raw_resolution, G.raw_resolution], device=device)
            raw_mask = torch.torch.empty([batch_gpu, 1, G.raw_resolution, G.raw_resolution], device=device)
            misc.print_module_summary(G, [{'image_raw': raw_image, 'image_mask': raw_mask}])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None:
            for param in misc.params_and_buffers(module):
                if param.numel() > 0 and num_gpus > 1:
                    torch.distributed.broadcast(param, src=0)

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs)  # subclass of training.loss.Loss
    phases = []
    if exp_cfg == 'raw-tex':
        for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
            if reg_interval is None:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
            else:  # Lazy regularization.
                mb_ratio = reg_interval / (reg_interval + 1)
                opt_kwargs = dnnlib.EasyDict(opt_kwargs)
                opt_kwargs.lr = opt_kwargs.lr * mb_ratio
                opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
                opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
                phases += [dnnlib.EasyDict(name=name + 'main', module=module, opt=opt, interval=1)]
                phases += [dnnlib.EasyDict(name=name + 'reg', module=module, opt=opt, interval=reg_interval)]
    else:
        opt = dnnlib.util.construct_class_by_name(params=G.parameters(), **G_opt_kwargs)  # subclass of torch.optim.Optimizer
        phases += [dnnlib.EasyDict(name='Gmain', module=G, opt=opt, interval=1)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, images_raw, masks, masks_raw = setup_snapshot_image_grid(training_set=training_set)
        if exp_cfg == 'raw-tex':
            save_image_grid(images_raw, os.path.join(run_dir, 'reals.png'), grid_size=grid_size, drange=None)
            save_image_grid(masks_raw, os.path.join(run_dir, 'masks.png'), grid_size=grid_size, drange=[0, 1])
            grid_z = torch.randn([images.shape[0], G.z_dim], device=device).split(batch_gpu)
        elif exp_cfg == 'res-tex':
            save_image_grid(images[:, G.raw_channels:], os.path.join(run_dir, 'reals.png'), grid_size=grid_size, drange=None)
            save_image_grid(masks, os.path.join(run_dir, 'masks.png'), grid_size=grid_size, drange=[0, 1])
            grid_image = torch.tensor(images[:, G.raw_channels:], dtype=torch.float32, device=device).split(batch_gpu)
            grid_mask = torch.tensor(masks, dtype=torch.float32, device=device).split(batch_gpu)
        elif exp_cfg == 'super-res':
            save_image_grid(images[:, :G.img_channels], os.path.join(run_dir, 'reals.png'), grid_size=grid_size, drange=None)
            save_image_grid(images_raw, os.path.join(run_dir, 'reals_raw.png'), grid_size=grid_size, drange=None)
            grid_image = torch.tensor(images_raw, dtype=torch.float32, device=device).split(batch_gpu)
            grid_mask = torch.tensor(masks_raw, dtype=torch.float32, device=device).split(batch_gpu)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_img_raw, phase_real_mask, phase_real_mask_raw = next(training_set_iterator)
            phase_real_img = phase_real_img.to(device).split(batch_gpu)
            phase_real_img_raw = phase_real_img_raw.to(device)
            phase_real_img_raw = phase_real_img_raw.split(batch_gpu)
            phase_real_mask = phase_real_mask.to(device).split(batch_gpu)
            phase_real_mask_raw = phase_real_mask_raw.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_z in zip(phases, all_gen_z):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_img_raw, real_mask, real_mask_raw, gen_z in zip(phase_real_img, phase_real_img_raw, phase_real_mask, phase_real_mask_raw, phase_gen_z):
                if exp_cfg == 'raw-tex':
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img_raw, real_mask=real_mask_raw, gen_z=gen_z, gain=phase.interval, cur_nimg=cur_nimg)
                elif exp_cfg == 'res-tex':
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_mask=real_mask, coords=coords, gain=phase.interval, cur_nimg=cur_nimg)
                elif exp_cfg == 'super-res':
                    loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_img_raw=real_img_raw, real_mask=real_mask, real_mask_raw=real_mask_raw, coords=coords,
                                              gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        if G_ema is not None:
            with torch.autograd.profiler.record_function('Gema'):
                ema_nimg = ema_kimg * 1000
                if ema_rampup is not None:
                    ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
                ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
                for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                    b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        # fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        # fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        # fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        # fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        # fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        stats_collector.update()
        for name, value in stats_collector.as_dict().items():
            if name.startswith("Loss/") and not "signs" in name:
                fields += [f"{name[5:]} {value.mean:<6.3f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            if exp_cfg == 'raw-tex':
                out = [G_ema(z=z, noise_mode='const') for z in grid_z]
            elif exp_cfg == 'res-tex':
                out = [G(img={'image': image, 'image_mask': mask}, noise_mode='const') for image, mask in zip(grid_image, grid_mask)]
            elif exp_cfg == 'super-res':
                out = [G(img={'image_raw': image_raw, 'image_mask': mask_raw}) for image_raw, mask_raw in zip(grid_image, grid_mask)]
            images = torch.cat([o['image'].cpu() for o in out]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), grid_size=grid_size, drange=None)
            if 'image_mask' in out[0]:
                images_mask = torch.cat([o['image_mask'].cpu() for o in out]).numpy()
                save_image_grid(images_mask, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}_mask.png'), grid_size=grid_size, drange=[0, 1])

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------
