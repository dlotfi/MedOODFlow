import csv
import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics


class OODEvaluator(BaseEvaluator):
    def __init__(self, config: Config):
        """OOD Evaluator.

        Args:
            config (Config): Config file from
        """
        super(OODEvaluator, self).__init__(config)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None

    def eval_ood(self,
                 net: nn.Module,
                 id_data_loaders: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor,
                 fsood: bool = False):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'test' in id_data_loaders, \
            'id_data_loaders should have the key: test!'
        dataset_name = self.config.dataset.name

        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)

        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loaders['test'])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)
        if self.config.recorder.get('save_tail_samples', False):
            self._save_tail_samples(id_data_loaders['test'], id_pred, id_conf,
                                    id_gt, dataset_name)

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders['csid'].items():
                print(f'Performing inference on {dataset_name} dataset...',
                      flush=True)
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt,
                                      dataset_name)
                if self.config.recorder.get('save_tail_samples', False):
                    self._save_tail_samples(csid_dl, csid_pred, csid_conf,
                                            csid_gt, dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])

        if 'ood_splits' in self.config.evaluator and \
           type(self.config.evaluator.ood_splits) is list:
            ood_splits = self.config.evaluator.ood_splits
        else:
            ood_splits = ['nearood', 'farood']

        print(f'Evaluating splits: {ood_splits}')

        ood_preds, ood_confs, ood_gts = [], [], []
        metrics_list = []
        for split in ood_splits:
            # load 'split' data and compute ood metrics
            print(u'\u2500' * 70, flush=True)
            split_ood_list, split_metrics = self._eval_ood(
                net, [id_pred, id_conf, id_gt],
                ood_data_loaders,
                postprocessor,
                ood_split=split)
            ood_preds.append(split_ood_list[0])
            ood_confs.append(split_ood_list[1])
            ood_gts.append(split_ood_list[2])
            metrics_list.append(split_metrics)

        print('Computing total macro-average metrics...', flush=True)
        metrics_list = np.concatenate(metrics_list, axis=0)
        metrics_mean = np.mean(metrics_list, axis=0)
        print('Computing total micro-average metrics...', flush=True)
        all_pred = np.concatenate([id_pred] + ood_preds)
        all_conf = np.concatenate([id_conf] + ood_confs)
        all_label = np.concatenate([id_gt] + ood_gts)
        ood_metrics = compute_all_metrics(all_conf, all_label, all_pred)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name='all (macro-average)')
            self._save_csv(ood_metrics, dataset_name='all (micro-average)')

    def _eval_ood(
            self,
            net: nn.Module,
            id_list: List[np.ndarray],
            ood_data_loaders: Dict[str, Dict[str, DataLoader]],
            postprocessor: BasePostprocessor,
            ood_split: str = 'nearood') -> Tuple[List[np.ndarray], np.ndarray]:
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        ood_split_pred, ood_split_conf, ood_split_gt = [], [], []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)
            if self.config.recorder.get('save_tail_samples', False):
                self._save_tail_samples(ood_dl, ood_pred, ood_conf, ood_gt,
                                        dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

            ood_split_pred.append(ood_pred)
            ood_split_conf.append(ood_conf)
            ood_split_gt.append(ood_gt)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)

        ood_split_pred = np.concatenate(ood_split_pred)
        ood_split_conf = np.concatenate(ood_split_conf)
        ood_split_gt = np.concatenate(ood_split_gt)
        return [ood_split_pred, ood_split_conf, ood_split_gt], metrics_list

    def eval_ood_val(self, net: nn.Module, id_data_loaders: Dict[str,
                                                                 DataLoader],
                     ood_data_loaders: Dict[str, DataLoader],
                     postprocessor: BasePostprocessor):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        assert 'val' in id_data_loaders
        assert 'val' in ood_data_loaders
        if self.config.postprocessor.APS_mode:
            val_auroc = self.hyperparam_search(net, id_data_loaders['val'],
                                               ood_data_loaders['val'],
                                               postprocessor)
        else:
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loaders['val'])
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loaders['val'])
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            val_auroc = ood_metrics[1]
        return {'auroc': 100 * val_auroc}

    def _save_csv(self, metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics

        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join(self.config.output_dir, 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join(self.config.output_dir, 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 fsood: bool = False,
                 csid_data_loaders: DataLoader = None):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()
        self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
            net, data_loader)

        if fsood:
            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                csid_pred, csid_conf, csid_gt = postprocessor.inference(
                    net, csid_dl)
                self.id_pred = np.concatenate([self.id_pred, csid_pred])
                self.id_conf = np.concatenate([self.id_conf, csid_conf])
                self.id_gt = np.concatenate([self.id_gt, csid_gt])

        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_data_loader,
        ood_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            id_pred, id_conf, id_gt = postprocessor.inference(
                net, id_data_loader)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, ood_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print('Hyperparam:{}, auroc:{}'.format(hyperparam,
                                                   aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))
        return max_auroc

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results

    def _save_tail_samples(self, dataloader: DataLoader, pred: np.ndarray,
                           conf: np.ndarray, gt: np.ndarray,
                           dataset_name: str):
        imglist = []
        for batch in dataloader:
            if 'image_name' not in batch:
                print(f"WARNING: 'image_name' not found in "
                      f'batch for {dataset_name}')
                return
            for i, name in enumerate(batch['image_name']):
                if 'image_path' in batch:
                    imglist.append((name, batch['image_path'][i]))
                else:
                    imglist.append((name, None))
        if len(imglist) != len(conf):
            print(f"WARNING: Number of images ({len(imglist)}) doesn't match "
                  f'number of scores ({len(conf)}) for {dataset_name}')
            return
        # Create rows with file path, confidence, prediction, and label
        rows = []
        for i, img in enumerate(imglist):
            rows.append({
                'image_name': img[0],
                'image_path': img[1],
                'score': float(conf[i]),
                'pred': int(pred[i]),
                'label': int(gt[i])
            })
        # Sort by confidence score
        rows.sort(key=lambda x: x['score'], reverse=True)
        # Save top and bottom 10%
        sample_count = len(rows)
        tail_size = max(int(sample_count * 0.1), 1)  # Ensure at least 1 sample
        # Create dataset-specific directory
        save_dir = os.path.join(self.config.output_dir, 'tail_samples',
                                dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        # Copy images and save top 10%
        top_tail = rows[:tail_size]
        output_path = os.path.join(save_dir, 'top_10p.csv')
        self._copy_images_and_save_csv(top_tail, output_path)
        # Copy images and save bottom 10%
        bottom_tail = rows[-tail_size:]
        bottom_tail.reverse()  # Reverse to have the lowest confidence first
        output_path = os.path.join(save_dir, 'bottom_10p.csv')
        self._copy_images_and_save_csv(bottom_tail, output_path)
        print(f'Saved top and bottom 10% ({tail_size} samples each) '
              f'for {dataset_name}')

    @staticmethod
    def _copy_images_and_save_csv(rows, csv_path):
        columns = [k for k in rows[0].keys() if k != 'image_path']
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows([{k: row[k] for k in columns} for row in rows])
        dst_dir = os.path.dirname(csv_path)
        if rows[0]['image_path'] is None:
            return
        for row in rows:
            image_name = os.path.basename(row['image_name'])
            shutil.copy2(row['image_path'], os.path.join(dst_dir, image_name))
