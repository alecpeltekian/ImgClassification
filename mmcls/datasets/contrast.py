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



@DATASETS.register_module()
class ContrastDataset(BaseDataset):

    # only one class for the lymph node
    CLASSES = ('PC', 'SYNTHCON')

    # get path to the csv file
    ffn_csvSavePath = 'data/renalData_new.csv'

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
                png_img = img[: img.find('.nii')] + '.png'
                if self.data_prefix.endswith('/'):
                  ffp_image = self.data_prefix + png_img
                else:
                  ffp_image = self.data_prefix + '/' + png_img

                img_info = {}
                img_info['gt_label'] = np.array(idx, dtype=np.int64)
                img_info['filename'] = ffp_image
                img_info['img_prefix'] = None
                data_infos.append(img_info)

        return data_infos 


    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

