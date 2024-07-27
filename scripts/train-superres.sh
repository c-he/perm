#!/bin/bash

python src/train.py --cfg=super-res --outdir=training-runs --data=data/neural-textures --gpus=1 --batch=4 --glr=0.002 --aug=noaug --snap=50 \
       --sr_mode=hybrid --lambda_tex=1 --lambda_geo=1 --lambda_reg=0.1 \
       --head_mesh=data/head.obj --roots=data/roots/rootPositions_10k.txt --strand_codec=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=10 --fft=true

# ablation study: output coeff directly
# python src/train.py --cfg=super-res --outdir=training-runs --data=data/neural-textures --gpus=1 --batch=4 --glr=0.002 --aug=noaug --snap=50 \
#        --sr_mode=coeff --lambda_tex=1 --lambda_geo=1 --lambda_reg=0 \
#        --head_mesh=data/head.obj --roots=data/roots/rootPositions_10k.txt --strand_codec=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=10 --fft=true