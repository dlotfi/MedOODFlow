#!/bin/bash
# sh scripts/ood/nflow/organamnist/organamnist_visualize.sh

SEED=0
MARK="final_feat"

# feature extraction
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_28x28.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_nflow_nflow_e100_lr0.0001_${MARK}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained True \
    --network.backbone.checkpoint "./results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.backbone.checkpoint_key "net" \
    --seed ${SEED} \
    --mark ${MARK}

# draw plots
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/pipelines/test/visualize_nflow_ood.yml \
    --visualizer.ood_scheme ood \
    --visualizer.score_dir "./results/organamnist_nflow_test_ood_ood_nflow_${MARK}/s${SEED}/ood/scores" \
    --visualizer.feat_dir "./results/organamnist_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --visualizer.ood_splits nearood farood \
    --visualizer.spectrum.types aggregate split \
    --visualizer.tsne_nflow.types aggregate split \
    --visualizer.tsne_score.types aggregate split \
    --seed ${SEED} \
    --mark ${MARK}
