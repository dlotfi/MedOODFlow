#!/bin/bash
# sh scripts/ood/nflow/organamnist/organamnist_visualize_typicality_raw.sh

SEED=0
MARK="raw"

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_identity.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_typicality_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --network.nflow.l2_normalize True \
    --network.nflow.hidden_size 2048 \
    --network.nflow.n_flows 32 \
    --network.nflow.clamp_value 1.0 \
    --seed ${SEED} \
    --mark ${MARK}

# draw plots
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/pipelines/test/visualize_nflow_ood.yml \
    --visualizer.ood_scheme ood \
    --visualizer.score_dir "./results/organamnist_nflow_test_nflow_ood_nflow_typicality_${MARK}/s${SEED}/ood/scores" \
    --visualizer.feat_dir "./results/organamnist_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --visualizer.ood_splits nearood farood \
    --visualizer.spectrum.types all splits \
    --visualizer.tsne_nflow.types all splits \
    --visualizer.tsne_nflow.l2_normalize_feat True \
    --visualizer.tsne_score.types all splits \
    --visualizer.tsne_score.l2_normalize_feat True \
    --seed ${SEED} \
    --mark ${MARK}
