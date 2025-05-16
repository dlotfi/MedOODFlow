import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Callable
from .tsne_visualizer import TSNEVisualizer
from openood.utils.vis_comm import save_fig_and_close, MARKERS


class TSNEScoreVisualizer(TSNEVisualizer):
    def draw_tsne_score_plot(self, feats_dict, scores_dict, title, output_path,
                             colored_id, id_splits, label_fn: Callable[[str],
                                                                       str]):
        log_scale = self.plot_config.score_log_scale
        fig_size = (float(self.plot_config.fig_size[0]),
                    float(self.plot_config.fig_size[1]))
        point_size = int(self.plot_config.point_size)
        no_title = self.plot_config.get('no_title', False)
        plt.figure(figsize=fig_size, dpi=300)
        tsne_feats_dict = TSNEVisualizer._tsne_compute(feats_dict)
        all_scores = np.concatenate(
            [scores for key, scores in scores_dict.items()])
        cmap = plt.cm.rainbow
        if log_scale:
            min_score = all_scores.min()
            if min_score <= 0:
                all_scores = all_scores + abs(min_score) + 1
            norm = mcolors.LogNorm(vmin=all_scores.min(),
                                   vmax=all_scores.max())
        else:
            norm = mcolors.Normalize(vmin=all_scores.min(),
                                     vmax=all_scores.max())
        marker_dict = {
            key: MARKERS[i % len(MARKERS)]
            for i, key in enumerate(feats_dict.keys())
        }
        for key, tsne_feats in tsne_feats_dict.items():
            scores = scores_dict[key]
            marker = marker_dict[key]
            if key in id_splits and not colored_id:
                plt.scatter(tsne_feats[:, 0],
                            tsne_feats[:, 1],
                            s=point_size,
                            alpha=0.2,
                            marker=marker,
                            label=label_fn(key),
                            c='grey')
            else:
                plt.scatter(tsne_feats[:, 0],
                            tsne_feats[:, 1],
                            s=point_size,
                            alpha=0.5,
                            marker=marker,
                            label=label_fn(key),
                            c=scores,
                            cmap=cmap,
                            norm=norm)
        plt.colorbar(mappable=plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=plt.gca())
        plt.xticks([])
        plt.yticks([])
        if not no_title:
            legend = plt.legend(loc='upper left', fontsize='small')
            for handle in legend.legend_handles:
                handle.set_color('black')
            plt.title(title)
        save_fig_and_close(output_path)

    def plot_tsne_score(self):
        output_dir = self.config.output_dir
        l2_normalize_feat = self.plot_config.l2_normalize_feat
        z_normalize_feat = self.plot_config.z_normalize_feat
        colored_id = self.plot_config.colored_id
        n_samples = self.plot_config.n_samples

        feats_dict = {}
        scores_dict = {}
        for split_name, dataset_list in self.datasets.items():
            feats = self.load_features([f'{d}.npz' for d in dataset_list],
                                       separate=True,
                                       l2_normalize=l2_normalize_feat,
                                       z_normalize=z_normalize_feat)
            scores = self.load_scores([f'{d}.npz' for d in dataset_list],
                                      separate=True)
            scores, feats = self.random_sample([scores, feats],
                                               array_names=dataset_list,
                                               n_samples=n_samples)
            scores, feats = self.remove_outliers(split_name, scores, feats)
            feats_dict[split_name] = feats
            scores_dict[split_name] = scores

        print('Plotting t-SNE for features', flush=True)
        title_suffix, file_suffix = self.get_title_and_file_suffix(
            l2_normalize_feat, z_normalize_feat)
        title = f't-SNE for{title_suffix} Backbone Features of ' \
                'ID and OOD Samples'
        output_path = os.path.join(output_dir, f'tsne_scores{file_suffix}.svg')
        self.draw_tsne_score_plot(feats_dict, scores_dict, title, output_path,
                                  colored_id, self.id_splits, self.get_label)

    def plot_tsne_score_split(self):
        output_dir = os.path.join(self.config.output_dir, 'split_plots')
        l2_normalize_feat = self.plot_config.l2_normalize_feat
        z_normalize_feat = self.plot_config.z_normalize_feat
        colored_id = self.plot_config.colored_id
        n_samples = self.plot_config.n_samples

        os.makedirs(output_dir, exist_ok=True)

        id_feats_dict = {}
        id_scores_dict = {}
        feats_dict = {}
        scores_dict = {}
        for split_name, dataset_list in self.datasets.items():
            for dataset in dataset_list:
                feats = self.load_features([f'{dataset}.npz'],
                                           separate=True,
                                           l2_normalize=l2_normalize_feat,
                                           z_normalize=z_normalize_feat)
                scores = self.load_scores([f'{dataset}.npz'], separate=True)
                scores, feats = self.random_sample([scores, feats],
                                                   array_names=[dataset],
                                                   n_samples=n_samples)
                if split_name in self.id_splits:
                    id_feats_dict[split_name] = feats
                    id_scores_dict[split_name] = scores
                else:
                    feats_dict.setdefault(split_name, {})[dataset] = feats
                    scores_dict.setdefault(split_name, {})[dataset] = scores

        # Plot t-SNE with scores for all pairs of 'id'
        # and one of the other datasets
        for split_name, datasets_feats in feats_dict.items():
            print(f'Plotting t-SNE for features with scores of {split_name}',
                  flush=True)
            combined_feats_dict = {**id_feats_dict, **datasets_feats}
            combined_scores_dict = {
                **id_scores_dict,
                **scores_dict[split_name]
            }
            title_suffix, file_suffix = self.get_title_and_file_suffix(
                l2_normalize_feat, z_normalize_feat)
            title = f't-SNE for{title_suffix} Backbone Features with ' \
                    f'Scores of ID and {split_name} Samples'
            output_path = os.path.join(
                output_dir,
                f'tsne_features{file_suffix}_scores_{split_name}.svg')
            self.draw_tsne_score_plot(combined_feats_dict,
                                      combined_scores_dict, title, output_path,
                                      colored_id, self.id_splits,
                                      self.get_dataset_label)

    def draw(self):
        if 'all' in self.plot_config.types:
            print(f'\n{" Overall ":-^50}', flush=True)
            self.plot_tsne_score()
        if 'splits' in self.plot_config.types:
            print(f'\n{" Splits ":-^50}', flush=True)
            self.plot_tsne_score_split()
