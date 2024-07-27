#!/bin/bash

python src/train.py --cfg=raw-tex --outdir=training-runs --data=data/neural-textures --gpus=1 --batch=4 --glr=0.002 --dlr=0.001 --gamma=5 --gamma-mask=1 --map-depth=4 --aug=noaug --snap=50 \
       --strand_codec=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=10 --fft=true
       