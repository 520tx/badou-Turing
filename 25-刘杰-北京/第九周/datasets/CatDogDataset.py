#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：CatDogDataset.py
@Author  ：luigi
@Date    ：2021/8/31 5:07 下午 
'''

from torch.utils.data import Dataset
import cv2, numpy as np
from glob import glob


class CatDogDataset(Dataset):

    def __init__(self, root_dir):
        paths = glob(root_dir)
        self.xs = np.array([])
        self.ys = np.array([])
        for path in paths:
            image = cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            x = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            x = x / 255
            # y = np.array((1, 0)) if 'cat' in path.split('/')[-1] else np.array((0, 1))
            y = 1 if 'cat' in path.split('/')[-1] else 0
            self.xs = np.append(self.xs, x)
            self.ys = np.append(self.ys, y)
        self.xs = self.xs.reshape((-1, 28 * 28))
        self.ys = self.ys.reshape((-1, 1))

    def __getitem__(self, ix):
        x, y = self.xs[ix], self.ys[ix]
        return x, y

    def __len__(self):
        return len(self.ys)




