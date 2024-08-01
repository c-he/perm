#!/bin/bash

python src/eval_blend_shapes.py --indir data/public-hair --outdir evaluation/fft-strands-pca/ --bsdir data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff 64 --save_strands true
python src/eval_blend_shapes.py --indir data/public-hair --outdir evaluation/strands-pca/ --bsdir data/blend-shapes/strands-blend-shapes.npz --n_coeff 64 --save_strands true