import numpy as np
import cv2
import os
from keras.utils import np_utils


def generate_dataset(root_dir, label_file, num_classes, train_ratio=0.7, shuffle=True,
                     size=(224, 224), transform=None):
    '''
    root_dir: 训练图片所在文件夹，
    label_file: 标注文件路径，其格式如下：
        image_name1,class_label1
        image_name2,class_label2
        .....
        image_nameN,class_labelN
    '''
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if shuffle:
        np.random.shuffle(lines)
    X_train, Y_train= [], []
    X_test, Y_test = [], []
    for line in lines:
        img_name, class_id= line.strip().split(',')
        img = cv2.imdecode(np.fromfile(os.path.join(root_dir, img_name), dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 预处理
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        img = img.astype(np.float32)
        img /= 255.

        if np.random.random() <= train_ratio:
            X_train.append(img)
            Y_train.append(class_id)
        else:
            X_test.append(img)
            Y_test.append(class_id)

    # 处理图像
    X_train = np.array(X_train).reshape(-1, 224, 224, 3)
    Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=num_classes)

    X_test = np.array(X_train).reshape(-1, 224, 224, 3)
    Y_test = np_utils.to_categorical(np.array(Y_train), num_classes=num_classes)

    return (X_train, Y_train), (X_test, Y_test)


def generate_dataset_batch(root_dir, label_file, batch_size, num_classes,
                           train_ratio=0.7, shuffle=True, size=(224, 224), transform=None):
    '''
        root_dir: 训练图片所在文件夹，
        label_file: 标注文件路径，其格式如下：
            image_name1,class_label1
            image_name2,class_label2
            .....
            image_nameN,class_labelN
    '''
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if shuffle:
        np.random.shuffle(lines)
    n = len(lines)
    train_num = int(train_ratio*len(lines))
    train_generator = batch_generator(root_dir, lines[:train_num], batch_size, num_classes, size, transform)
    val_generator = batch_generator(root_dir, lines[train_num:], batch_size, num_classes, size, transform)
    return train_generator, train_num, val_generator, (n-train_num)


def batch_generator(root_dir, lines, batch_size, num_classes, size=(224, 224), transform=None):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            img_name, class_id = lines[i].strip().split(';')
            img = cv2.imdecode(np.fromfile(os.path.join(root_dir, img_name), dtype=np.uint8), 1)

            # 预处理
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            img = img.astype(np.float32)
            img /= 255.

            X_train.append(img)
            Y_train.append(class_id)
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = np.array(X_train).reshape(-1, 224, 224, 3)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=num_classes)
        yield (X_train, Y_train)



def center_crop(img):
    # 将图片修剪成中心的正方形
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img


def resize_img(img, size):
    return cv2.resize(img, size)


def normalize_img(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img = img.astype(np.float32)
    img /= 255.
    if mean is not None and std is not None:
        img[0, :, :] -= mean[0]
        img[1, :, :] -= mean[1]
        img[2, :, :] -= mean[2]
        img[0, :, :] /= std[0]
        img[1, :, :] /= std[1]
        img[2, :, :] /= std[2]
    return img


