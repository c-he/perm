#!/bin/bash

# augment dataset by horizontally flipping each hairstyle
python src/preprocess.py --process_fn=flip --indir=data/usc-hair --outdir=data/usc-hair
python src/preprocess.py --process_fn=flip --indir=data/usc-hair-mix --outdir=data/usc-hair-mix

# solve blend shapes for strands (FFT produces better bases to preserve curvature)
python src/preprocess.py --process_fn=blend_shapes --indir=data/usc-hair --outdir=data/blend-shapes --bs_type=strands --n_coeff=64 --svd_solver=full --fft=true

# fit neural textures with 64 PCA coefficients (nearest interpolation produces better results than bilinear when sampled with different hair roots)
# 686 (343 x 2) hairstyles from USC-HairSalon
# 20,368 (10,184 x 2) augmented hairstyles using HairMix
python src/preprocess.py --process_fn=texture --indir=data/usc-hair --outdir=data/neural-textures/high-res --size=256 --interp_mode=nearest --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=64 --fft=true
python src/preprocess.py --process_fn=texture --indir=data/usc-hair-mix --outdir=data/neural-textures/high-res --size=256 --interp_mode=nearest --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=64 --fft=true

# compress fitted neural textures to obtain textures for guide strands (we use 10 PCA coefficients to smooth guide strands)
python src/preprocess.py --process_fn=guide_strands --indir=data/neural-textures/high-res --outdir=data/neural-textures/low-res --size=32 --interp_mode=nearest --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --n_coeff=10 --fft=true

# compute mean and std for strand PCA coefficients
python src/preprocess.py --process_fn=normalize --indir=data/neural-textures/high-res --outdir=data/neural-textures

# transform strands to the canonical space defined by each guide strand
# python src/preprocess.py --process_fn=canonical --indir=data/neural-textures --outdir=data/usc-hair-canonical --interp_mode=nearest --bsdir=data/blend-shapes/fft-strands-blend-shapes.npz --fft=true