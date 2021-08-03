import mmcv
import numpy as np

import pandas as pd
import cv2

import copy
import os
import os.path as osp
import nibabel as nib
from scipy import ndimage

from .base_dataset import BaseDataset
from .builder import DATASETS
from sklearn.metrics import precision_recall_fscore_support
from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy



@DATASETS.register_module()
class ContrastDataset(BaseDataset):

    # only one class for the lymph node
    CLASSES = ('PC', 'SYNTHCON')

    # get path to the csv file
    ffn_csvSavePath = '/home/alec/renalData_new.csv'

    # read the csv file  
    df = pd.read_csv(ffn_csvSavePath)

    def load_annotations(self):

        # convert the LN label to a number
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}

        t_ann_file = os.path.basename(self.ann_file)
        
        # get the right data split from the CSV file        
        if t_ann_file == 'train.txt' or t_ann_file == 'train':
            split_str = 'train'
        elif t_ann_file == 'val.txt' or t_ann_file == 'valid.txt' or t_ann_file == 'valid' or t_ann_file == 'val':
            split_str = 'valid'
        elif t_ann_file == 'test.txt' or t_ann_file == 'test':
            split_str = 'test'


        # get the rows that correspond to the data split
        df_split = self.df[self.df['subset'] == split_str]    
        
        print('num images:', len(df_split))
        
        data_infos = []

        for row in df_split.iterrows():
            pc_img = self.CLASSES[0] + '/img/' + row[1][self.CLASSES[0]]
            pc_num = row[1][self.CLASSES[0] + '_num']
            syn_img = self.CLASSES[1] + '/img/' +  row[1][self.CLASSES[1]]
            syn_num = row[1][self.CLASSES[1] + '_num']

            # create Custom Middle dataset format
            for idx, (img, n_img) in enumerate(zip([pc_img, syn_img], [pc_num, syn_num])):
              for k in range(n_img):
                # get image file path
                png_img = img[: img.find('.nii')] + f'_{k}.png'
                if self.data_prefix.endswith('/'):
                  ffp_image = self.data_prefix + png_img
                else:
                  ffp_image = self.data_prefix + '/' + png_img

                img_info = {}
                img_info['gt_label'] = np.array(idx, dtype=np.int64)
                img_info['img_info'] = {'filename': ffp_image}
                img_info['img_prefix'] = None
                data_infos.append(img_info)

        return data_infos 


    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metirc {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if average_mode != 'micro':
          if 'support' in metrics:
              support_value = support(
                  results, gt_labels, average_mode=average_mode)
              eval_results['support'] = support_value

          precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
          if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
              if thrs is not None:
                  precision_recall_f1_values = precision_recall_f1(
                      results, gt_labels, average_mode=average_mode, thrs=thrs)
              else:
                  precision_recall_f1_values = precision_recall_f1(
                      results, gt_labels, average_mode=average_mode)
              for key, values in zip(precision_recall_f1_keys,
                                    precision_recall_f1_values):
                  if key in metrics:
                      if isinstance(thrs, tuple):
                          eval_results.update({
                              f'{key}_thr_{thr:.2f}': value
                              for thr, value in zip(thrs, values)
                          })
                      else:
                          eval_results[key] = values

        elif average_mode == 'micro':
          pred_label = np.argsort(results, axis=1)[:, -1]
          p, r, f, _ = precision_recall_fscore_support(gt_labels, pred_label, average=average_mode)
          eval_results.update({
            'Precision': p*100,
            'Recall': r*100,
            'F1-score': f*100
          })

        return eval_results
