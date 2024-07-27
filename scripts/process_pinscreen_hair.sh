#!/bin/bash
# resample all hairstyles to have ~10k strands
python src/preprocess.py --process resample --input data/pinscreen --output /data/pinscreen_resample --roots data/roots/rootPositions_10k.txt

# decompose hairstyles into guide strands and local strands
python src/preprocess.py --process hair --input data/pinscreen --output data/hair --n_clusters 100 --smooth
python src/preprocess.py --process hair --input data/pinscreen_resample/data --output data/hair_resample --n_clusters 100 --smooth

# solve blend shapes for each hair component (FFT produces better bases to preserve the curvature)
python src/preprocess.py --process pca --input data/hair --output data/blend_shapes/strands --blend_shape_type strand --pca_components 128 --svd_solver full

# fit neural textures for each hair component (nearest sampling produces better results than bilinear sampling)
# number of blend shapes for Pinscreen hair data: 64 for original and local strands, 32 for guide strands
python src/preprocess.py --process texture --input data/hair_resample --output data/neural_textures --texture_type strand --interp_mode nearest
# python src/preprocess.py --process texture --input data/hair_resample --output data/neural_textures --texture_type strand_pca --interp_mode nearest --blend_shapes data/blend_shapes/strands/strands_blend_shapes.npz --num_coeff 64
# python src/preprocess.py --process texture --input data/hair_resample --output data/neural_textures --texture_type guide --interp_mode nearest --blend_shapes data/blend_shapes/strands/guide_strands_blend_shapes.npz --num_coeff 32
# python src/preprocess.py --process texture --input data/hair_resample --output data/neural_textures --texture_type local --interp_mode nearest --blend_shapes data/blend_shapes/strands/local_strands_blend_shapes.npz --num_coeff 64

python src/preprocess.py --process texture --input data/hair_resample --output data/neural_textures --texture_type strand_pca --interp_mode nearest --blend_shapes data/blend_shapes/strands_fft/strands_blend_shapes.npz --num_coeff 64 --fft
python src/preprocess.py --process texture --input data/hair_resample --output data/neural_textures --texture_type guide --interp_mode nearest --blend_shapes data/blend_shapes/strands_fft/guide_strands_blend_shapes.npz --num_coeff 32 --fft
python src/preprocess.py --process texture --input data/hair_resample --output data/neural_textures --texture_type local --interp_mode nearest --blend_shapes data/blend_shapes/strands_fft/local_strands_blend_shapes.npz --num_coeff 64 --fft