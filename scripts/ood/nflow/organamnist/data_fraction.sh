#!/bin/bash
# sh scripts/ood/nflow/organamnist/data_fraction.sh

SEED=0
DATA_PATH="data/benchmark_imglist/medmnist"

python tools/random_subset.py ${DATA_PATH}/train_organamnist.txt 10 ${DATA_PATH}/train_organamnist_10p.txt 1754
python tools/random_subset.py ${DATA_PATH}/train_organamnist.txt 25 ${DATA_PATH}/train_organamnist_25p.txt 1754
python tools/random_subset.py ${DATA_PATH}/train_organamnist.txt 50 ${DATA_PATH}/train_organamnist_50p.txt 1754
