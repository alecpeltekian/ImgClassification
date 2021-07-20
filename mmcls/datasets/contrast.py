import mmcv
import numpy as np

import pandas as pd
import cv2

import copy
import os
import os.path as osp
import nibabel as nib

from .base_dataset import BaseDataset
from .builder import DATASETS



@DATASETS.register_module()
class ContrastDataset(BaseDataset):

    # only one class for the lymph node
    CLASSES = ('PC', 'SYNTHCON')

    # get path to the csv file
    ffn_csvSavePath = '/content/mmclassification/data/renalData.csv'

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
        
        print('num images:', len(df_split))
        
        data_infos = []

        for row in df_split.iterrows():
            pc_img = row[1][CLASSES[0]]
            syn_img = row[1][CLASSES[1]]

            # create Custom Middle dataset format
            for idx, img in enumerate([pc_img, syn_img]):
              data_info = {}
              
              # get image file path
              if self.data_prefix.endswith('/'):
                ffp_image = self.data_prefix + img
              else:
                ffp_image = self.data_prefix + '/' + img

              # populate
              norm_img = process_scan(ffp_image)
              img_width, img_height, img_depth = norm_img.shape
              for k in range(img_depth):
                img_info = {}
                img_info['gt_label'] = np.array([idx], dtype=np.int64)
                slice_img = norm_img[:, :, k]
                img_info['img'] = np.repeat(np.expand_dims(slice_img, axis=-1), 3, axis=-1)
                data_infos.append(img_info)

        return data_infos 


    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
      
      
def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan
  
  
def normalize(volume, thres_norm=(-1000, 600)):
    """Normalize the volume"""
    min, max = thres_norm
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img, desired_shape=(224, 224, 64), keep_depth=True):
    """Resize across z-axis"""
    # Set the desired depth
    desired_width, desired_height, desired_depth = desired_shape

    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    if keep_depth:
      img = ndimage.zoom(img, (width_factor, height_factor, 1), order=1)
    else:
      img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume
