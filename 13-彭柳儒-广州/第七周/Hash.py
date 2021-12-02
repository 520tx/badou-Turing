import cv2
import numpy as np


def hammingDistance(hash1, hash2):
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    #         hamming_distance = 0
    #         s = str(bin(x^y))      #内置函数 bin() 的作用是将输入的十进制数，转换成二进制
    #         for i in range(2,len(s)):
    #             if int(s[i]) is 1:
    #                 hamming_distance += 1
    hamming_distance = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            hamming_distance += 1
    return hamming_distance


def aHash(img):
    re_img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_LINEAR)
    h, w = re_img.shape[0:2]
    hash_var = []
    for y in range(h):
        for x in range(w):
            mean = np.mean(re_img[y, :])
            var = re_img[y, x]
            # print("all:{},mean:{},var:{}".format(re_img[y, :], mean, var))
            if var > mean:
                hash_var.append(1)
            else:
                hash_var.append(0)
    hash_strNums = [str(x) for x in hash_var]
    hash_str = ''.join(hash_strNums)
    hash_16 = hex(int(hash_str, 2))  # 2转16
    return hash_var


def pHash(img):
    re_img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_LINEAR)
    h, w = re_img.shape[0:2]
    # print("h,w:",h, w)
    hash_var = []
    for y in range(h):
        for x in range(1, w):
            # print("all:{},mean:{},var:{}".format(re_img[y, :], mean, var))
            # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
            if re_img[y, x - 1] > re_img[y, x] :
                hash_var.append(1)
            else:
                hash_var.append(0)
    hash_strNums = [str(x) for x in hash_var]
    hash_str = ''.join(hash_strNums)
    hash_16 = hex(int(hash_str, 2))  # 2转16
    return hash_var

img_1 = cv2.imread('lenna.png',0)
img_2 = cv2.imread('lenna_noise.png',0)

img_1_hash = aHash(img_1)
img_2_hash = aHash(img_2)
print(img_1_hash)
print(img_2_hash)
n = hammingDistance(img_1_hash, img_2_hash)
print('均值哈希算法相似度：',n)

img_1_hash = pHash(img_1)
img_2_hash = pHash(img_2)
print(img_1_hash)
print(img_2_hash)
n = hammingDistance(img_1_hash, img_2_hash)
print('差值哈希算法相似度：',n)