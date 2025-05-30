#!/bin/bash
# sh scripts/ood/she/organmnist_test_ood_nci.sh

PYTHONPATH='.':$PYTHONPATH \

python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_28x28.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/nci.yml \
    --num_workers 8 \
    --network.checkpoint "./results/organamnist_resnet18_28x28/s0/resnet18_28_1.pth" \
    --network.checkpoint_key "net"
