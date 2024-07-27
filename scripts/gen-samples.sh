#!/bin/bash

python src/gen_samples.py --outdir=./out-samples/test --model=hair_models/parahair_v1.0 --roots=data/roots/rootPositions_10k.txt --head_mesh=data/head.obj --seeds=0-99 --trunc=0.8