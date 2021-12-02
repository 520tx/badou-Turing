#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：hash_manual.py
@Author  ：luigi
@Date    ：2021/8/3 下午7:14 
'''

import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from 第六周 import noise_generator
from imutils import paths


def ahash(gray):
    """ 均值hash的实现

    :param gray: 灰度图
    :type gray: np.ndarray
    :return: 哈希列表
    :rtype: np.ndarray
    """
    gray = cv2.resize(gray, (8, 8), interpolation = cv2.INTER_CUBIC)
    mean = np.mean(gray)
    result = np.where(gray < mean, 0, 1).reshape(64,)
    return result


def dHash(gray):
    """插值hash的实现

    :param gray: 灰度图
    :type gray: np.ndarray
    :return: 哈希列表
    :rtype: np.ndarray
    """
    gray = cv2.resize(gray, (9, 8), interpolation = cv2.INTER_CUBIC)
    gray1 = gray[:, 0:8]
    gray2 = cv2.resize(gray, (9, 8))[:, 1:]
    result = np.where(gray1 < gray2, 0, 1).reshape(64,)
    return result

def cmpHash(hash1, hash2):
    """ 对比hash

    :param hash1: 图像1对应的hash
    :type hash1: np.ndarray
    :param hash2: 图像2对应的hash
    :type hash2: np.ndarray
    :return: 汉明距离
    :rtype: int
    """
    return np.sum(hash1 != hash2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p","--path",required=True, help="path for input image")
    args = vars(ap.parse_args())
    imagePaths = np.array(list(paths.list_images(args["path"])))
    random = np.random.choice(imagePaths.shape[0], 2, replace=False)

    image = cv2.imread(imagePaths[random[0]])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_cmp = noise_generator.add_pepper_salt(gray, 0.2)

    image_cmp = cv2.imread(imagePaths[random[1]])
    gray_cmp = cv2.cvtColor(image_cmp, cv2.COLOR_BGR2GRAY)

    ah = ahash(gray)
    ah_cmp = ahash(gray_cmp)
    dh = ahash(gray)
    dh_cmp = ahash(gray_cmp)
    result1 = cmpHash(ah, ah_cmp)
    result2 = cmpHash(dh, dh_cmp)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.subplot(221)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(image_cmp,cv2.COLOR_BGR2RGB))
    plt.text(-600, 800, '均值hash：{}'.format(result1))
    plt.text(-600, 1000, '差值hash：{}'.format(result2))
    plt.show()

if __name__ == '__main__':
    main()








