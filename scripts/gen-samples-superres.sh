#!/bin/bash

python src/gen_samples_superres.py --sr_mode=nearest --network=hair_models/perm_v1.0/unet-superres.pkl --source=data/test --outdir=evaluation/superres/nearest --head_mesh=data/head.obj
python src/gen_samples_superres.py --sr_mode=bilinear --network=hair_models/perm_v1.0/unet-superres.pkl --source=data/test --outdir=evaluation/superres/bilinear --head_mesh=data/head.obj
python src/gen_samples_superres.py --sr_mode=neural --network=hair_models/perm_v1.0/unet-superres.pkl --source=data/test --outdir=evaluation/superres/unet-weight --head_mesh=data/head.obj
python src/gen_samples_superres.py --sr_mode=neural --network=evaluation/models/unet-superres-coeff.pkl --source=data/test --outdir=evaluation/superres/unet-coeff --head_mesh=data/head.obj