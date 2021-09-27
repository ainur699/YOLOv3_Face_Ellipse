﻿import os
import math
import numpy as np
import cv2
import glob
import torch
import torch.utils.data as data


class VoxCelebDataset(data.Dataset):

    def __init__(self, root_dir, is_train):
        self.root_dir = root_dir

        if is_train:
            self.videos = glob.glob(os.path.join(self.root_dir, 'train/*/*.png'))
        else:
            self.videos = glob.glob(os.path.join(self.root_dir, 'test/*/*.png'))

    def load_img(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise Exception('None Image')

        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255

        return img

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        image = self.load_img(self.videos[idx])
        
        out = {}
        out['image'] = image
        out['path'] = os.path.join(*os.path.normpath(self.videos[idx]).split(os.sep)[-2:])
        return out