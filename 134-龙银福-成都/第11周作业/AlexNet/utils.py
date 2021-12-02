import matplotlib.image as mpimg
import cv2
import tensorflow as tf
import numpy as np

# 将图片修剪成中心的正方形
def load_image(path):
    img = mpimg.imread(path) # 读取图片 (rgb)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

# 改变 image 的尺寸大小为 size 大小的图像
def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

# 通过预测的数值结果 输出 对应的文字类别
def print_answer(argmax):
    with open("./data/model/index_word.txt", "r", encoding='utf-8') as f:
        # f.readlines() --> 取出文件中所有数据存放在一个列表，列表中的每一个元素代表文件中的一行
        # l.split(";") 通过指定分隔符 ";" 对字符串进行切片
        synset = [l.split(";")[1][:-1] for l in f.readlines()]

    print(synset[argmax])
    return synset[argmax]

