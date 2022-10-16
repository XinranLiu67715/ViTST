import torch
from torch.utils.data import Dataset
import os
import random
from image import *
import numpy as np
import numbers
from torchvision import datasets, transforms
import torch.nn.functional as F
import json
import torchvision.transforms.functional as Fi



class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4, args=None):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        fname = self.lines[index]['fname']
        img = self.lines[index]['img']
        gt_count = self.lines[index]['gt_count']

        '''data augmention'''
        if self.train == True:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        gt_count = gt_count.copy()
        img = img.copy()

        if self.train == True:
            if self.transform is not None:
                img = self.transform(img)

            return fname, img, gt_count

        else:
            if self.transform is not None:
                img = self.transform(img)

            width, height = img.shape[2], img.shape[1]

            m = int(width / 384)
            n = int(height / 384)
            for i in range(0, m):
                for j in range(0, n):

                    if i == 0 and j == 0:
                        img_return = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                    else:
                        crop_img = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                        img_return = torch.cat([img_return, crop_img], 0).cuda()
            return fname, img_return, gt_count

class my_data(Dataset):
    def __init__(self, root_path, output_path, dataset, traits, label_path=None, predict_key=None, augmentation=True,
                 aug_pro=0.5, resize_shape=0, rotate_angle=[0, 0, 0, 0], phase='training'):
        super(my_data, self).__init__()
        if label_path is None:
            self.label_file = None
        else:
            with open(label_path, 'r') as f:
                self.label_file = json.load(f)['Measurements']
        self.output_path = output_path
        self.dataset = dataset
        self.image_path = os.path.join(root_path)
        self.predict_key = predict_key
        self.aug_func_id = [0, 1, 2]
        self.aug_func = [Fi.vflip, Fi.hflip, Fi.rotate]
        self.aug_pro = aug_pro
        self.resize_scale = resize_shape
        self.rotate_angle = rotate_angle
        self.augmentation = augmentation
        self.trait = traits
        self.phase = phase

    def __getitem__(self, index):
        # image extraction and precessing
        image_read_path = os.path.join(self.image_path, self.dataset[index])

        print('self.image_path', self.image_path, 'self.dataset[index]', self.dataset[index], 'index', index)
        if os.path.exists(image_read_path):
            print(image_read_path)
            with open(image_read_path, 'rb') as f:
                rgb_image = Image.open(f)
                rgb_image.load()

            if self.augmentation:
                if torch.rand(1) > self.aug_pro:
                    transform_func_id = random.choice(self.aug_func_id)
                    transform_func = self.aug_func[transform_func_id]
                    if transform_func_id == 0 or transform_func_id == 1:
                        rgb_image = transform_func(rgb_image)

                    else:
                        rotate_angle = random.choice(self.rotate_angle)
                        rgb_image = transform_func(rgb_image, rotate_angle)

            img = np.array(rgb_image).astype(np.float32)
            img = transforms.ToTensor()(img)

            key_num = self.dataset[index].split('.')[0].split('_')[1]
            if self.phase == 'training' :
                gt_rec = []
                key = "Image" + str(key_num)
                measurements = self.label_file
                quantity = measurements[key]['quantity']
                gt_rec.append(quantity)
                gt_rec = np.array(gt_rec).reshape(-1, 1).astype(np.float32)
                return img, gt_rec, key_num
            else:
                return img, key_num

    def __len__(self):
        return len(self.dataset)