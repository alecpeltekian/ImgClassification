import csv
import os
import random
import cv2
from scipy import ndimage
import nibabel as nib
import numpy as np
import pandas as pd
import copy


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume, bound=(-374, 426)):
    """Normalize the volume"""
    volume = np.clip(volume, bound[0], bound[1]) 
    volume = 255*((volume - bound[0]) / (bound[1] - bound[0]))
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired shape
    desired_width = 224
    desired_height = 224

    # Get current depth
    current_width = img.shape[0]
    current_height = img.shape[1]

    # Compute depth factor
    width = current_width / desired_width
    height = current_height / desired_height

    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, 1), order=1)
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


cur_csv_file = '/home/alec/renalData.csv'
new_csv_file = '/home/alec/renalData_new.csv'
path_dir = '/mnt/cadlabnas/datasets/RenalDonors'

data = pd.read_csv(cur_csv_file)       
CLASSES = ['PC', 'SYNTHCON']
os.makedirs(f'{path_dir}/{CLASSES[0]}/img', exist_ok=True)
os.makedirs(f'{path_dir}/{CLASSES[1]}/img', exist_ok=True)

with open(new_csv_file, 'w') as f:
  writer = csv.DictWriter(f, fieldnames=['PC', 'PC_num', 'SYNTHCON', 'SYNTHCON_num', 'subset'])
  writer.writeheader()
  for row in data.iterrows():
    pc_img = row[1][CLASSES[0]]
    syn_img = row[1][CLASSES[1]]
    row_data = {}
    row_data['subset'] = row[1]['subset']
    for folder, filename in zip(CLASSES, [pc_img, syn_img]):
      if filename.endswith('.nii.gz'):
        full_path_filename = f'{path_dir}/{folder}/{filename}'
        scan = process_scan(full_path_filename)
        n_images = scan.shape[0]
        row_data[folder + '_num'] = n_images
        row_data[folder] = filename

        # for a nifti file, dim_0 is image width, dim_1 is height, dim_2 is channels/slices
        im_width, im_height, nifti_channels = np.asarray(scan).shape
        im_channels = 3
        half_c = im_channels // 3
        for i in range(n_images):
          # check bounds of data
          # within
          if (i - half_c >= 0) & (i + half_c <= nifti_channels - half_c):
              slice_start = i - half_c
              slice_end = i + half_c + 1
              # get temporary sub_slice
              t_sub_slice = scan[:,:,slice_start:slice_end]

          # outside left edge
          elif i - half_c < 0:
              # get temporary sub_slice
              t_sub_slice = scan[:,:,0:im_channels]

          # outside right edge
          elif i + half_c > nifti_channels - 1:
              # get temporary sub_slice
              t_sub_slice = scan[:,:,nifti_channels-im_channels:nifti_channels]


          # current shape of subslice is [width, height, channels]
          # need to change shape to [height, width, channels] for saving to disk/training
          curr_sub_slice = np.zeros([im_height, im_width, im_channels], dtype = np.uint8)

          # histogram equalization (adaptive with CLAHE)
          # run histEq on the normalized_9x_percentile midSlice
          ksize = 8
          clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(ksize, ksize))
          for channel_idx in range(im_channels):
              # transpose individual slice  ----- [width, height] --> [height, width]
              curr_slice = np.transpose(t_sub_slice[:,:,channel_idx])

              # normalize the slice
              # norm_9x_img = min_max_normalization(curr_slice, rquantile = 1)
              histEq_img = clahe.apply(np.asarray(curr_slice).astype(np.uint8)) 

              # place in position
              curr_sub_slice[:,:,channel_idx] = copy.deepcopy(histEq_img)

          # opencv saves images in BGR format (not RGB -- so interchange first and last images in sub-slice)
          tempRed = copy.deepcopy(curr_sub_slice[:,:,0])
          curr_sub_slice[:,:,0] = copy.deepcopy(curr_sub_slice[:,:,2])
          curr_sub_slice[:,:,2] = copy.deepcopy(tempRed)
          cv2.imwrite(f'{path_dir}/{folder}/img/{filename[:filename.find(".")]}_{i}.png', curr_sub_slice)
  
    writer.writerow(row_data)
