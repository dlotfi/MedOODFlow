#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_test_fsood_nflow_typicality.sh

SEED=0
MARK="final_feat"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --pipeline.extract_nflow False \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_typicality_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark ${MARK}

# evaluation
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/postprocessors/nflow_typicality.yml \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --dataset.feat_root "./results/brats20_t1_resnet3d_18_feat_extract_nflow_${MARK}/s${SEED}" \
    --ood_dataset.feat_root "./results/brats20_t1_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_typicality_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --seed ${SEED} \
    --mark ${MARK}
