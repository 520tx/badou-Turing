import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

pic_path = './lenna.png'
img = plt.imread(pic_path)
if pic_path[-4:] == '.png':
    img = img * 255
img = img.mean(axis=-1)
#1、高斯平滑
#sigma = 1.52  #高斯平滑时的高斯核参数
sigma = 0.5
dim = int(np.round(6 * sigma + 1))
if dim % 2 == 0:
    dim += 1

Guassian_filter = np.zeros([dim, dim])
print("Guassian", Guassian_filter)
tmp = [i-dim//2 for i in range(dim)]
print("tmp", tmp)
n1 = 1/(2*math.pi*sigma**2)
n2 = -1/(2*sigma**2)
for i in range(dim):
    for j in range(dim):
        Guassian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
Guassian_filter = Guassian_filter / Guassian_filter.sum()
print("Guassian", Guassian_filter)
dx, dy = img.shape
img_new = np.zeros(img.shape)
tmp = dim//2
img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_new[i, j] = np.sum(img_pad[i:i+dim, j:j+dim]*Guassian_filter)
plt.figure(1)
plt.imshow(img_new.astype(np.uint8), cmap='gray')
plt.axis('off')
# 2、求梯度。以下两个是滤波求梯度用的sobel矩阵（检测图像中的水平、垂直和对角边缘）
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_tidu_x = np.zeros(img_new.shape)
img_tidu_y = np.zeros([dx, dy])
img_tidu = np.zeros(img_new.shape)
img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
for i in range(dx):
    for j in range(dy):
        img_tidu_x[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_x)
        img_tidu_y[i, j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_kernel_y)
        img_tidu[i, j] = np.sqrt(img_tidu_x[i, j]**2 + img_tidu_y[i, j]**2)
img_tidu_x[img_tidu_x == 0] = 0.00000001
angle = img_tidu_y/img_tidu_x
plt.figure(2)
plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
plt.axis('off')

#3、非极大值抑制
for i 