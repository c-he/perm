#!/bin/bash

python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=64 --save_strands=false
python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=56 --save_strands=false
python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=48 --save_strands=false
python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=40 --save_strands=false
python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=32 --save_strands=false
python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=24 --save_strands=false
python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=16 --save_strands=false
python src/eval_blend_shapes.py --indir=data/public-hair --outdir=evaluation/fft-strands-coeff/ --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=8 --save_strands=false