import csv
import os
import random
import cv2
from scipy import ndimage
import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd


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
    # Convert (width, height, depth) -> (depth, width, height)
    volume = volume.transpose((2, 0, 1))
    # Add channels (depth, width, height) -> (depth, width, height, 3)
    volume = np.repeat(np.expand_dims(volume, axis=-1), 3, axis=-1)
    return volume


cur_csv_file = 'data/renalData.csv'    # Shoud change this
new_csv_file = 'data/renalData_new.csv'     # Shoud change this
path_dir = 'data/RenalDonors'        # Should change this

data = pd.read_csv(cur_csv_file)       
CLASSES = ['PC', 'SYNTHCON']
os.makedirs(f'{path_dir}/{CLASSES[0]}/img', exist_ok=True)
os.makedirs(f'{path_dir}/{CLASSES[1]}/img', exist_ok=True)

with open(new_csv_file, 'w') as f:
  writer = csv.DictWriter(f, fieldnames=['PC_img', 'PC_num', 'SYNTHCON_img', 'SYNTHCON_num'])
  writer.writeheader()
  for row in data.iterrows():
    pc_img = row[1][CLASSES[0]]
    syn_img = row[1][CLASSES[1]]
    row_data = {}
    for folder, filename in zip(CLASSES, [pc_img, syn_img]):
      if filename.endswith('.nii.gz'):
        full_path_filename = f'{path_dir}/{folder}/{filename}'
        scan = process_scan(full_path_filename)
        n_images = scan.shape[0]
        row_data[folder + '_num'] = n_images
        row_data[folder + '_img'] = filename
        for i in range(n_images):
          cv2.imwrite(f'{path_dir}/{folder}/img/{filename[:filename.find(".")]}_{i}.png', scan[i])

    writer.writerow(row_data)
