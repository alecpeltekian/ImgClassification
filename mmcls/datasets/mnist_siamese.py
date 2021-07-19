import mmcv
import numpy as np

import pandas as pd
import cv2

import copy
import os
import os.path as osp

from .base_dataset import BaseDataset
from .builder import DATASETS
from mmcls.core.evaluation import mse



@DATASETS.register_module()
class ExampleDataset(BaseDataset):

    # only one class for the lymph node
    CLASSES = ('NEG', 'POS')

    # get path to the csv file -- updated after removing Grace's corrections, incorrect RECIST, and to-be-reannotated RECIST images
    ffn_csvSavePath = '/home/alec/Desktop/ImgClassification/data/latin.csv'

    # read the csv file  
    df = pd.read_csv(ffn_csvSavePath)

    def load_annotations(self):
        ###
            # the ann_file is just a dummy name -- it identifies the right data split to choose from the csv file
        ###

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
        
        # read the dataframe column
        pd_col2read = 'real'

        # get the unique names of images that have GT annotations 
        uniq_strs = df_split[pd_col2read].unique()

        print('data split:', split_str)
        print('num unique images:', len(uniq_strs))
        
        data_infos = []

        for row in df_split.iterrows():
            fake_img = row[1]['fake']
            real_img = row[1]['real']

            # create Custom Middle dataset format
            data_info = {}
            
            for idx, img in enumerate([real_img, fake_img]):
              # get image file path
              if not img.endswith('.png'):
                img += '.png'
              
              if self.data_prefix.endswith('/'):
                ffp_image = self.data_prefix + img
              else:
                ffp_image = self.data_prefix + '/' + img

              # populate
              data_info[f'filename{idx+1}'] = ffp_image

              if idx == 0:
                data_infos.append({'img_info': {'filename1': ffp_image, 'filename2': ffp_image}, 'gt_label': np.array(1, dtype=np.int64), 'img_prefix': None})
              elif idx == 1:
                data_infos.append({'img_info': data_info, 'gt_label': np.array(0, dtype=np.int64), 'img_prefix': None})

        return data_infos 


    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    def evaluate(self,
                 results,
                 metric='mse',
                 metric_options=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `mse`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'mse',
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mse' in metrics:
            mse_value = mse(
                results, gt_labels)
            eval_results['mse'] = mse_value

        return eval_results
