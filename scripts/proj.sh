#!/bin/bash

python src/projector.py --model=hair_models/perm_v1.0 --target=data/neural-textures-hairnet/00.npz --num-steps-warmup=1000 --num-steps=9000 --outdir=fitting/hairnet-long-opt --head_mesh=data/head.obj --save-video=false