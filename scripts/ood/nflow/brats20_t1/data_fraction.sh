#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/data_fraction.sh

SEED=0
DATA_PATH="data/benchmark_imglist/medood"

python tools/random_subset.py ${DATA_PATH}/train_brats20_t1.txt 10 ${DATA_PATH}/train_brats20_t1_10p.txt 1754
python tools/random_subset.py ${DATA_PATH}/train_brats20_t1.txt 25 ${DATA_PATH}/train_brats20_t1_25p.txt 1754
python tools/random_subset.py ${DATA_PATH}/train_brats20_t1.txt 50 ${DATA_PATH}/train_brats20_t1_50p.txt 1754
