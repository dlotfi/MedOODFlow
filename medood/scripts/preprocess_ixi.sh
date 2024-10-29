#!/bin/bash

source ./scripts/common_env.sh

python preprocess_ixi.py \
    --base_dir="$RAW_DATASETS_DIR/IXI/images" \
    --output_dir="$PROCESSED_DATASETS_DIR/ixi_t1/" \
    --num_samples=250 \
    --seed=$SEED \
    --use_gpu \
    --skip_existing

python generate_imglist.py \
    --input_dir="$PROCESSED_DATASETS_DIR/ixi_t1/" \
    --base_dir="$PROCESSED_DATASETS_DIR" \
    --output_dir="$IMGLIST_DIR"
