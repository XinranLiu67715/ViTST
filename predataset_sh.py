import glob
import math
import os
import torch
import cv2
import h5py
import numpy as np
import scipy.io as io
import scipy.spatial
from scipy.ndimage.filters import gaussian_filter
import random
import json
import pdb
'''set your data path'''
root = '/home/yrx/lxr/ViTST/data/wfan'

part_train = os.path.join(root, 'train/')
part_test = os.path.join(root, 'testing/')

path_sets = [part_train, part_test]
print ('path_sets',path_sets)

'''for part A'''
if not os.path.exists(part_train.replace('images', 'gt_density_map_crop')):
    os.makedirs(part_train.replace('images', 'gt_density_map_crop'))

if not os.path.exists(part_test.replace('images', 'gt_density_map_crop')):
    os.makedirs(part_test.replace('images', 'gt_density_map_crop'))




img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.JPG')):
        img_paths.append(img_path)

img_paths.sort()

np.random.seed(0)
random.seed(0)
for img_path in img_paths:
    print('img_path',img_path)
    Img_data = cv2.imread(img_path)

    key_num = img_path.split('/')[8].split('_')[1].split('.')[0]
    key = "Image" + str(key_num)
    print(key_num)
    gt_json = os.path.join(root, 'json/zhangjiakou.json')
    label_path = gt_json
    with open(label_path, 'r') as f:
            label_file = json.load(f)['Measurements']
    measurements = label_file
    quantity = measurements[key]['quantity']


    save_path= []
    save_fname = 'IMG_' + str() + key_num 
    h5_path = root +'/gt_density_map_crop/' +save_fname +'.h5'

    print('h5_path',h5_path)

    with h5py.File(h5_path, 'w') as hf:
        print ('quantity',quantity)
        hf['gt_count'] = quantity 



    