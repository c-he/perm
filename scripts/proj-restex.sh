#!/bin/bash

python src/projector_restex.py --network=hair_models/parahair_v1.0/vae-res-texture.pkl --target=data/neural-textures-public-hair --outdir=evaluation/restex-vae-new --head_mesh=data/head.obj --num-steps=5000 --save-video=false
# python src/projector_restex.py --network=evaluation/models/stylegan2-res-texture.pkl --target=data/neural-textures-public-hair --outdir=evaluation/restex-stylegan --head_mesh=data/head.obj --num-steps=500 --save-video=false --reload_modules=true