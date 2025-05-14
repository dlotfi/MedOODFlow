#!/bin/bash
# sh scripts/ood/nflow/organamnist/organamnist_train_nflow_typicality_raw.sh

SEED=0
MARK="raw"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/identity.yml \
    configs/pipelines/train/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained False \
    --seed ${SEED} \
    --mark ${MARK}

# train
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/networks/nflow_identity.yml \
    configs/pipelines/train/train_nflow_typicality.yml \
    configs/postprocessors/nflow_typicality.yml \
    --dataset.feat_root "./results/organamnist_identity_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.nflow.l2_normalize True \
    --network.nflow.hidden_size 2048 \
    --network.nflow.n_flows 32 \
    --network.nflow.clamp_value 1.0 \
    --optimizer.grad_regularizer_lambda 10 \
    --recorder.save_last_model True \
    --seed ${SEED} \
    --mark ${MARK}
