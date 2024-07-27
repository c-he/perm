#!/bin/bash

python src/calc_metrics.py --network=$1 --hair-data=data/usc-hair-resample/npz --tex-data=data/neural-textures/pca-strands --hair-prop=roots,strands --max-size=1 --head-mesh=data/head.obj --hair-type=original --gpus=1