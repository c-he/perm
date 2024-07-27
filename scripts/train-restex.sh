#!/bin/bash

python src/train.py --cfg=res-tex --outdir=training-runs --data=data/neural-textures --gpus=1 --batch=4 --glr=0.002 --map-depth=4 --aug=noaug --snap=50 \
       --encode_space=w --lambda_tex=10 --lambda_geo=1 --lambda_kl=1e-4 \
       --head_mesh=data/head.obj --roots=data/roots/rootPositions_10k.txt --strand_codec=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=64 --fft=true