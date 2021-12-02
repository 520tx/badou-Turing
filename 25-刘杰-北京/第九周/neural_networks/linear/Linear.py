#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：Linear.py
@Author  ：luigi
@Date    ：2021/8/27 下午4:41 
'''
import numpy as np

class Liner():

    def __init__(self, in_features, out_features, bias=True):

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        if self.bias:
            self.weights = np.random.randn(self.in_features + 1, self.out_features)
        else:
            self.weights = np.random.randn(self.in_features, self.out_features)

    def forward(self, input):

        batch_size = input.shape[0]
        input = input / np.linalg.norm(input)
        X = np.hstack((input, np.ones((batch_size, 1)))) if self.bias else input
        return np.dot(X, self.weights)

    def grad(self, x):
        return x

    def __call__(self, input):
        return self.forward(input)
