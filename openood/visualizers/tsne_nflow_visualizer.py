import os

from .tsne_visualizer import TSNEVisualizer


class TSNEFlowVisualizer(TSNEVisualizer):
    def plot_tsne(self):
        output_dir = self.config.output_dir
        l2_normalize_feat = self.plot_config.l2_normalize_feat
        z_normalize_feat = self.plot_config.z_normalize_feat
        n_samples = self.plot_config.n_samples

        feats_dict = {}
        feats_flow_dict = {}
        for split_name, dataset_list in self.datasets.items():
            feats = self.load_features([f'{d}.npz' for d in dataset_list],
                                       separate=True,
                                       l2_normalize=l2_normalize_feat,
                                       z_normalize=z_normalize_feat)
            feats_flow = self.load_features(
                [f'{d}_flow.npz' for d in dataset_list], separate=True)
            feats, feats_flow = self.random_sample([feats, feats_flow],
                                                   array_names=dataset_list,
                                                   n_samples=n_samples)
            feats_dict[split_name] = feats
            feats_flow_dict[split_name] = feats_flow

        # Add extra feature files if specified
        extra_feats_dict = self.load_extra_features(n_samples)
        feats_dict.update(extra_feats_dict)

        # Get feature_splits from config if available
        feature_splits = getattr(self.plot_config, 'feature_splits', [])
        feature_splits = [int(size) for size in feature_splits]
        # Handle regular features like TSNEVisualizer
        if len(feature_splits) > 0:
            split_feats_list = self._split_features(feats_dict, feature_splits)
            for i, split_feats in enumerate(split_feats_list):
                self.plot_tsne_features(split_feats,
                                        l2_normalize_feat,
                                        z_normalize_feat,
                                        output_dir,
                                        segment_index=i + 1)
        # Original behavior for full features
        self.plot_tsne_features(feats_dict, l2_normalize_feat,
                                z_normalize_feat, output_dir)

        # Handle flow features with original logic
        print('Plotting t-SNE for normalizing flow features', flush=True)
        self.draw_tsne_plot(
            feats_flow_dict,
            't-SNE for Normalizing Flow Features of ID and OOD Samples',
            os.path.join(output_dir,
                         'tsne_features_flow.svg'), self.get_dataset_label)
