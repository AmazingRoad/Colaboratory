import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np

class GTSRBloader(data.Dataset):

    def __init__(self, num_class=43, im_size=48, txt_path='./data/train.txt', dataset_name='GTSRB'):
        self.txt_path = txt_path
        self.name = dataset_name
        self.num_class = num_class
        self.im_size = im_size
        self.data = []

        self.read_data()

    def __getitem__(self, index):
        im_path = self.data[index].strip().split(' ')[0]
        label = int(self.data[index].strip().split(' ')[1])

        img = cv2.imread(im_path).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.im_size, self.im_size))
        img = img[:,:, np.newaxis]

        return torch.from_numpy(img).permute(2, 0, 1), label

    def pull_image(self, index):
        im_path = self.data[index].strip().split(' ')[0]
        label = int(self.data[index].strip().split(' ')[1])

        img = cv2.imread(im_path).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (self.im_size, self.im_size))
        img = img[:,:, np.newaxis]

        return torch.from_numpy(img).permute(2, 0, 1), label

    def __len__(self):
        return len(self.data)

    def read_data(self):
        with open(self.txt_path, 'r') as f:
            self.data = f.readlines()
            print('Total {} images...'.format(len(self.data)))
