#!/bin/bash
# sh scripts/ood/nflow/organamnist/organamnist_test_ood_nflow_raw.sh

SEED=0
MARK="raw"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_identity.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_${MARK}/s${SEED}/epoch-last_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --network.nflow.l2_normalize True \
    --network.nflow.hidden_size 2048 \
    --network.nflow.n_flows 32 \
    --network.nflow.clamp_value 1.0 \
    --seed ${SEED} \
    --mark ${MARK}

# evaluation
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_identity.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/postprocessors/nflow.yml \
    --dataset.feat_root "./results/organamnist_identity_feat_extract_nflow_${MARK}/s${SEED}" \
    --ood_dataset.feat_root "./results/organamnist_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_${MARK}/s${SEED}/epoch-last_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --network.nflow.l2_normalize True \
    --network.nflow.hidden_size 2048 \
    --network.nflow.n_flows 32 \
    --network.nflow.clamp_value 1.0 \
    --seed ${SEED} \
    --mark ${MARK}
