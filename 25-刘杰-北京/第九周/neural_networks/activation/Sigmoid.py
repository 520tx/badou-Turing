#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：Sigmoid.py
@Author  ：luigi
@Date    ：2021/9/2 2:11 下午 
'''
import numpy as np


class Sigmoid():

    def __call__(self, input):
        return self.forward(input)

    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def grad(self, x):
        fn_x = self.forward(x)
        return fn_x * (1 - fn_x)
