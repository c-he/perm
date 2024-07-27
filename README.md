<p align="center">

  <h2 align="center">Perm: A Parametric Representation for Multi-Style 3D Hair Modeling</h2>
  <p align="center">
    <a href="https://xavierchen34.github.io/"><strong>Chengan He</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=JYVCn3AAAAAJ&hl=en"><strong>Xin Sun</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=8zksQb4AAAAJ&hl=zh-CN"><strong>Zhixin Shu</strong></a>
    ·
    <a href="https://shenyujun.github.io/"><strong>Fujun Luan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=7LhjCn0AAAAJ&hl=en"><strong>Soren Pirk</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Jorge Alejandro Amador Herrera</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Dominik L. Michels</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Tuanfeng Y. Wang</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Meng Zhang</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Holly Rushmeier</strong></a>
    ·
    <a href="https://hszhao.github.io/"><strong>Yi Zhou</strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2307.09481"><img src='https://img.shields.io/badge/arXiv-Perm-red' alt='Paper PDF'></a>
        <a href='https://ali-vilab.github.io/AnyDoor-Page/'><img src='https://img.shields.io/badge/Project_Page-Perm-green' alt='Project Page'></a>
    <br>
    <b>The University of Hong Kong &nbsp; | &nbsp;  Alibaba Group  | &nbsp;  Ant Group </b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="assets/Figures/Teaser.png">
    </td>
    </tr>
  </table>

## Getting Started

Before installing Perm, make sure you have CUDA Toolkit 11.3 installed as noted in EG3D/StyleGAN3 (https://github.com/NVlabs/eg3d/tree/main#requirements), which would be required to compile customized CUDA ops.
CUDA Toolkit 11.3 has been tested on my machine with Ubuntu 22.04, which also requires `gcc <= 10` (I use `gcc=10.5.0`). To install perm, run:

```bash
bash scripts/setup.sh
```

### TODO

- [ ] Release cleaned codebase.
- [ ] Release pre-trained checkpoints.
- [ ] Release augmented USC-HairSalon that contains ~20k hairstyles.
- [ ] Release fitted Perm parameters for the original 343 hairstyles in USC-HairSalon.

### Data Processing

#### Strand Resampling

First, we need to use **Stylist** to resample all hair models to make sure each strand has 100 sample points. In stylist directory, run:
```bash
bash scripts/do-the-job.sh
```

#### Processing hair data necessary for Perm

In this directory, run:
```bash
bash scripts/process-usc-hair.sh
```
This script will:
1. horizontally flip each hair model to augment the dataset.
2. solve PCA blend shapes for hair strands.
3. fit neural textures with PCA coefficients (nearest interpolation produces better results than bilinear when sampled with different hair roots).
4. compress neural textures from `256x256` to `32x32` to obtain textures for **guide strands**.

### StyleGAN Training

In this directory, run:
```bash
bash scripts/train.sh
```
This script will train a StyleGAN2 as our parametric model for the fitted neural textures.

## Citation

If you found this code or paper useful, please consider citing:
```bibtex
@inproceedings{he2022nemf,
    author = {He, Chengan and Saito, Jun and Zachary, James and Rushmeier, Holly and Zhou, Yi},
    title = {NeMF: Neural Motion Fields for Kinematic Animation},
    booktitle = {NeurIPS},
    year = {2024}
}
```

## Contact

If you run into any problems or have questions, please create an issue or contact `chengan.he@yale.edu`. To obtain models trained on Adobe's internal data, please reach out to `yizho@adobe.com` for an individual usage license.