<p align="center">

  <h2 align="center">Perm: A Parametric Representation for Multi-Style 3D Hair Modeling</h2>
  <p align="center">
    <a href="https://cs.yale.edu/homes/che/"><strong>Chengan He<sup>1</sup></strong></a>
    ·
    <a href="https://www.sunxin.name/"><strong>Xin Sun<sup>2</sup></strong></a>
    ·
    <a href="https://zhixinshu.github.io/"><strong>Zhixin Shu<sup>2</sup></strong></a>
    ·
    <a href="https://luanfujun.com/"><strong>Fujun Luan<sup>2</sup></strong></a>
    ·
    <a href="https://storage.googleapis.com/pirk.io/index.html"><strong>Sören Pirk<sup>3</sup></strong></a>
    ·
    <a href="https://cemse.kaust.edu.sa/people/person/jorge-alejandro-amador-herrera"><strong>Jorge Alejandro Amador Herrera<sup>4</sup></strong></a>
    ·
    <a href="http://dmichels.de/"><strong>Dominik L. Michels<sup>4</sup></strong></a>
    ·
    <a href="https://tuanfeng.github.io/"><strong>Tuanfeng Y. Wang<sup>2</sup></strong></a>
    ·
    <a href="https://mengzephyr.com/"><strong>Meng Zhang<sup>5</sup></strong></a>
    ·
    <a href="https://graphics.cs.yale.edu/people/holly-rushmeier"><strong>Holly Rushmeier<sup>1</sup></strong></a>
    ·
    <a href="https://zhouyisjtu.github.io/"><strong>Yi Zhou<sup>2</sup></strong></a>
    <br>
    <br>
        <a href="https://arxiv.org/abs/2407.19451"><img src="https://img.shields.io/badge/arXiv-2407.19451-b31b1b" height=22.5 alt='Paper PDF'></a>
        <a href='https://cs.yale.edu/homes/che/projects/perm/'><img src="https://img.shields.io/badge/Project_Page-perm-green" height=22.5 alt='Project Page'></a>
    <br>
    <b><sup>1</sup> Yale University &nbsp; | &nbsp; <sup>2</sup> Adobe Research &nbsp; | &nbsp; <sup>3</sup> CAU &nbsp; | &nbsp; <sup>4</sup> KAUST &nbsp; | &nbsp; <sup>5</sup> Nanjing University of Science and Technology </b>
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="perm.png">
    </td>
    </tr>
  </table>

## TODO

- [ ] Release cleaned codebase.
- [ ] Release pre-trained checkpoints.
- [ ] Release augmented USC-HairSalon that contains ~20k hairstyles.
- [ ] Release fitted Perm parameters for the original 343 hairstyles in USC-HairSalon.
- [ ] Release checkpoints trained on more curly data (v2).
- [ ] Release a reimplementation of our single-view reconstruction pipeline with a public license.

## Getting started

### Installation

Before installing perm, make sure you have CUDA Toolkit 11.3 (or later) installed as noted in [EG3D/StyleGAN3](https://github.com/NVlabs/eg3d/tree/main#requirements), which would be required to compile customized CUDA ops.
CUDA Toolkit 11.3 has been tested on my machine with Ubuntu 22.04, which also requires `gcc <= 10` (I use `gcc=10.5.0`). 
You can use the following commands with Miniconda3 to create and activate your Python environment:

```bash
conda env create -f environment.yml
conda activate perm
pip install -r requirements.txt --no-cache-dir
```

### USC-HairSalon

Since the [original link](http://www-scf.usc.edu/~liwenhu/SHM/database.html) of USC-HairSalon has been deprecated for a while, you can obtain a copy of it from this [Google Drive link](https://drive.google.com/file/d/118ZwW_pDw9IvnoTndHMk4wLZcPb0cw4v/view). We augment these data using the style mixing algorithm described in [HairNet](https://github.com/papagina/HairNet_DataSetGeneration) to enlarge the dataset size to ~10k hairstyles and make sure each strand has 100 sample points.

### Data Processing

#### Strand Resampling

First, we need to use **Stylist** to resample all hair models to make sure each strand has 100 sample points. In stylist directory, run:
```bash
conda env create -f environment.yml
conda activate perm
pip install -r requirements.txt --no-cache-dir
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

## Training

In this directory, run:
```bash
bash scripts/train.sh
```
This script will train a StyleGAN2 as our parametric model for the fitted neural textures.

## Rendering

Most of the figures in our paper are rendered using [Hair Tool](https://joseconseco.github.io/HairTool_3_Documentation/) in Blender. We highly recommend checking out this excellent addon!

## Acknowledgements

- Our code structure is based on [EG3D](https://github.com/NVlabs/eg3d).
- Our naming convention and model formulation are heavily influenced by [SMPL](https://smpl.is.tue.mpg.de/).

**Huge thanks to these great open-source projects!**

## Citation

If you found this code or paper useful, please consider citing:
```bibtex
@article{he2024perm,
    title={Perm: A Parametric Representation for Multi-Style 3D Hair Modeling},
    author={Chengan He and Xin Sun and Zhixin Shu and Fujun Luan and S\"{o}ren Pirk and Jorge Alejandro Amador Herrera and Dominik L. Michels and Tuanfeng Y. Wang and Meng Zhang and Holly Rushmeier and Yi Zhou},
    journal={arXiv preprint arXiv::2407.19451},
    year={2024}
}
```

## Contact

If you run into any problems or have questions, please create an issue or contact `chengan.he@yale.edu`. To obtain models trained on Adobe's internal data and our single-view reconstruction pipeline, please reach out to `yizho@adobe.com` for an **individual release license**.