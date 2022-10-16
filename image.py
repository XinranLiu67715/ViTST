import scipy.spatial
from PIL import Image
import scipy.io as io
import scipy
import numpy as np
import h5py
import pdb
def load_data(img_path, args, train=True):
    key_num = img_path.split('/')[8].split('.')[0].split('_')[1]
    gt_path = './data/gt_density_map_crop/'+'IMG_'+str(key_num)+'.h5'

    img = Image.open(img_path).convert('RGB')


    while True:
        try:
            gt_file = h5py.File(gt_path)
            gt_count = np.asarray(gt_file['gt_count'])
            break  # Success!
        except OSError:
            print("load error:", img_path)
            break

    img = img.copy()
    gt_count = gt_count.copy()

    return img, gt_count
