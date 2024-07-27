#!/bin/bash

conda env create -f environment.yml
conda activate doris
pip install -r requirements.txt --no-cache-dir