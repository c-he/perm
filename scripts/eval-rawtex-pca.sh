#!/bin/bash

python src/eval_rawtex_pca.py --eval_mode=recon --indir=data/neural-textures-public-hair/low-res --outdir=evaluation/guide-strands-pca --tex_bsdir=data/blend-shapes/raw-tex-blend-shapes.npz --strand_bsdir=data/blend-shapes/fft-strands-blend-shapes.npz \
       --head_mesh=data/head.obj --save_strands=true


# python src/eval_rawtex_pca.py --eval_mode=sample --seeds=0-99 --indir=data/neural-textures-public-hair/low-res --outdir=evaluation/guide-strands-pca --tex_bsdir=data/blend-shapes/raw-tex-blend-shapes.npz --strand_bsdir=data/blend-shapes/fft-strands-blend-shapes.npz \
#        --head_mesh=data/head.obj --save_strands=true