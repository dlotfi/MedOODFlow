import os
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from .base_visualizer import BaseVisualizer
from openood.utils.vis_comm import save_fig_and_close


class SpectrumVisualizer(BaseVisualizer):
    @staticmethod
    def get_optimal_bins(data):
        """Determine optimal number of bins using Freedman-Diaconis rule.

        Formula: bin width = 2 * IQR(x) / (n^(1/3))
        where IQR is the interquartile range and n is number of observations.
        """
        if len(data) == 0:
            return 10  # Default for empty data

        # Calculate bin width using Freedman-Diaconis rule
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        n = len(data)

        # Handle potential issues
        if iqr == 0:
            # Fall back to Sturges' rule if IQR is 0
            return int(np.ceil(np.log2(n) + 1))

        bin_width = 2 * iqr / (n**(1 / 3))
        if bin_width == 0:
            return int(np.sqrt(n))  # Simple fallback

        data_range = np.max(data) - np.min(data)
        n_bins = int(np.ceil(data_range / bin_width))

        # Cap at reasonable limits
        return max(min(n_bins, 100), 5)

    def draw_histogram(self, scores_dict, output_path, title,
                       label_fn: Callable[[str], str]):
        log_scale = self.plot_config.score_log_scale
        n_bins = self.plot_config.n_bins
        fig_size = (float(self.plot_config.fig_size[0]),
                    float(self.plot_config.fig_size[1]))
        no_title = self.plot_config.get('no_title', False)
        if n_bins == 'auto':
            # Combine all data to determine optimal bins for consistent view
            all_scores = np.concatenate(list(scores_dict.values()))
            n_bins = SpectrumVisualizer.get_optimal_bins(all_scores)
            print(f'Optimal number of bins: {n_bins}', flush=True)
        else:
            n_bins = int(n_bins)

        plt.figure(figsize=fig_size, dpi=300)
        for key, scores in scores_dict.items():
            plt.hist(scores,
                     n_bins,
                     density=True,
                     alpha=0.5,
                     label=label_fn(key),
                     log=log_scale)
        plt.yticks([])
        # Make background of the plot transparent
        # ax = plt.gca()
        # ax.patch.set_alpha(0.0)
        # fig = plt.gcf()
        # fig.patch.set_alpha(0.0)
        if not no_title:
            plt.legend(loc='upper left', fontsize='small')
            plt.title(title)
        save_fig_and_close(output_path)

    def plot_spectrum(self):
        output_dir = self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        scores_dict = {}
        for split_name, dataset_list in self.datasets.items():
            scores = self.load_scores([f'{d}.npz' for d in dataset_list])
            scores = self.remove_outliers(split_name, scores)
            scores_dict[split_name] = scores

        print('Plotting log-likelihood histogram', flush=True)
        self.draw_histogram(scores_dict,
                            os.path.join(output_dir, 'spectrum.svg'),
                            'Log-Likelihood for ID and OOD Samples',
                            self.get_label)

    def plot_spectrum_split(self):
        output_dir = os.path.join(self.config.output_dir, 'split_plots')
        os.makedirs(output_dir, exist_ok=True)

        id_scores_dict = {}
        scores_dict = {}
        for split_name, dataset_list in self.datasets.items():
            for dataset in dataset_list:
                scores = self.load_scores([f'{dataset}.npz'])
                scores = self.remove_outliers(dataset, scores)
                if split_name in self.id_splits:
                    id_scores_dict[split_name] = scores
                else:
                    scores_dict.setdefault(split_name, {})[dataset] = scores

        # Plot each split against ID
        for split_name, datasets_scores in scores_dict.items():
            print(f'Plotting log-likelihood histogram for {split_name}',
                  flush=True)
            combined_scores_dict = {**id_scores_dict, **datasets_scores}
            self.draw_histogram(
                combined_scores_dict,
                os.path.join(output_dir, f'spectrum_{split_name}.svg'),
                f'Log-Likelihood for ID and {split_name} Samples',
                self.get_dataset_label)

    def plot_spectrum_dataset(self):
        output_dir = os.path.join(self.config.output_dir, 'dataset_plots')
        os.makedirs(output_dir, exist_ok=True)

        # Get all ID scores
        id_scores_dict = {}
        for split_name in self.id_splits:
            dataset_list = self.datasets[split_name]
            scores = self.load_scores([f'{d}.npz' for d in dataset_list])
            scores = self.remove_outliers(split_name, scores)
            id_scores_dict[split_name] = scores

        # Plot each dataset against ID
        for split_name, dataset_list in self.datasets.items():
            if split_name in self.id_splits:
                continue
            for dataset in dataset_list:
                print(
                    f'Plotting log-likelihood histograms for '
                    f'{dataset} dataset of {split_name}',
                    flush=True)
                scores = self.load_scores([f'{dataset}.npz'])
                scores = self.remove_outliers(dataset, scores)
                combined_scores_dict = {**id_scores_dict, dataset: scores}
                self.draw_histogram(
                    combined_scores_dict,
                    os.path.join(output_dir, f'spectrum_{dataset}.svg'),
                    f'Log-Likelihood for ID and {dataset} Samples',
                    self.get_dataset_label)

    def draw(self):
        if 'all' in self.plot_config.types:
            print(f'\n{" Overall ":-^50}', flush=True)
            self.plot_spectrum()
        if 'splits' in self.plot_config.types:
            print(f'\n{" Splits ":-^50}', flush=True)
            self.plot_spectrum_split()
        if 'datasets' in self.plot_config.types:
            print(f'\n{" Datasets ":-^50}', flush=True)
            self.plot_spectrum_dataset()
