#!/bin/bash

# python src/style_mixing.py --model hair_models/perm_v1.0 --hair1 fitting/usc-hair/npz/strands00270.npz --hair2 fitting/usc-hair/npz/strands00153.npz --steps 5 --outdir test_new --head_mesh hair_models/head.obj --interp_mode full

# python src/style_mixing.py --model hair_models/perm_v1.0 --hair1 evaluation/style-mixing-data/strands00270.npz --hair2 evaluation/style-mixing-data/strands00153.npz --steps 5 --outdir test_new --head_mesh hair_models/head.obj --interp_mode full

# python src/style_mixing.py --model hair_models/perm_v1.0 --hair1 fitting/hairnet-long-opt/00/00.npz --hair2 fitting/hairnet-long-opt/06/06.npz --steps 5 --outdir test_new_long_opt --head_mesh hair_models/head.obj --interp_mode full

# python src/style_mixing.py --model hair_models/perm_v1.0 --hair1 fitting/usc-hair/npz/strands00031.npz --hair2 fitting/usc-hair/npz/strands00343.npz --steps 5 --outdir test_new --head_mesh hair_models/head.obj --interp_mode theta

# python src/style_mixing.py --model hair_models/perm_v1.0 --hair1 fitting/usc-hair/npz/strands00002.npz --hair2 fitting/usc-hair/npz/strands00035.npz --steps 5 --outdir test_new --head_mesh hair_models/head.obj --interp_mode beta

python src/style_mixing.py --model hair_models/perm_v1.0 --hair1 fitting/usc-hair/npz/strands00035.npz --hair2 fitting/usc-hair/npz/strands00393.npz --steps 5 --outdir test_new --head_mesh hair_models/head.obj --interp_mode full