#!/bin/bash

python src/gen_samples_rawtex.py --outdir=out --head_mesh=data/head.obj --seeds=0-4 --trunc=0.7 --network=$1