#!/bin/bash

python preprocess_brats23_ped.py \
    --base_dir="$RAW_DATASETS_DIR/BraTS_2023/BraTS-PED/" \
    --output_dir="$PROCESSED_DATASETS_DIR/brats23_ped_t1/" \
    --num_samples=99 \
    --seed=328131023

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/brats23_ped_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
