#!/bin/bash

python src/projector_rawtex.py --network=hair_models/perm_v1.0/stylegan2-raw-texture.pkl --target=data/neural-textures-public-hair/low-res --outdir=evaluation/guide-strands-stylegan --head_mesh=data/head.obj --num-steps=5000 --save-video=false