#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：Relu.py
@Author  ：luigi
@Date    ：2021/9/1 4:50 下午 
'''
import numpy as np


class Relu():

    def __call__(self, input):
        return self.forward(input)

    def forward(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)
