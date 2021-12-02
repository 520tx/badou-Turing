# -------------------------------------------------------------#
#          InceptionV3的网络部分
# -------------------------------------------------------------#
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Input, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image


# 卷积层 + BatchNormalization层 + ReLU激活层
# 输入x，输出特征通道数filters，卷积核大小(num_row, num_col)，步长(1, 1)，默认填充方式使得输入和输出的h、w相等
def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    if name is not None:
        conv_name = name + '_conv'
        bn_name = name + '_bn'
    else:
        conv_name = None
        bn_name = None
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


# InceptionV3网络模型
# 输入尺寸(299, 299, 3)，输出类别数1000
def InceptionV3(input_shape=[299, 299, 3], classes=1000):
    img_input = Input(shape=input_shape)

    # [1] 第一个卷积层：输入(299, 299, 3)，输出(149, 149, 32)
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    # [2] 第二个卷积层：输入(149, 149, 32)，输出(147, 147, 32)
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    # [3] 第三个卷积层：输入(147, 147, 32)，输出(147, 147, 64)
    x = conv2d_bn(x, 64, 3, 3)
    # [4] 第一个最大池化层：池化窗口大小(3, 3)，输入(147, 147, 64)，输出(73, 73, 64)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # [5] 第四个卷积层：输入(73, 73, 64)，输出(71, 71, 80)
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    # [6] 第五个卷积层：输入(71, 71, 80)，输出(71, 71, 192)
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    # [7] 第二个最大池化层：池化窗口(3, 3)，输入(71, 71, 192)，输出(35, 35, 192)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # --------------------------------#
    # [8] Inception Block1 35x35
    #     3个Inception module(or part)
    # (35, 35, 192) -> (35, 35, 288)
    # --------------------------------#

    # [8-1] Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    # 64+64+96+32 = 256
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed0')

    # [8-2] Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 64+64+96+64 = 288
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed1')

    # [8-3] Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    # 64+64+96+64 = 288
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2')

    # --------------------------------#
    # [9] Inception Block2 17x17
    #   5个Inception module(or part)
    #   (35,35,288) -> (17,17,768)
    # --------------------------------#

    # [9-1] Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 384 + 96 + 288 = 768
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # [9-2] Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 192 * 4 = 768
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4')

    # [9-3]->[9-4] Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # 192 * 4 = 768
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i))

    # [9-5] Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

    # 192 * 4 = 768
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    # --------------------------------#
    # [10] Inception Block3 8x8
    #   3个Inception module(or part)
    #   (17x17x768)->(8x8x2048)
    # --------------------------------#

    # [10-1] Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 320 + 192 + 768 = 1280
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # [10-2]->[10-3] Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        # 384 * 2 = 768
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        # 384 * 2 = 768
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # 320 + 768 + 768 + 192 = 2048
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i))

    # [11] 全局平均池化 (8,8,2048)->(1,1,2048)
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    # [12] 全连接层 (1,1,2048)->(1,1,1000)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input
    # 实例化网络模型
    model = Model(inputs, x, name='inception_v3')

    return model


# 归一化处理 [-1, 1]
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    # [1] 实例化模型
    model = InceptionV3()
    # [2] keras构建深度学习模型，可以通过model.summary()输出模型各层的参数状况
    model.summary()
    # [3] 加载预训练的权重
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    # [4] 加载需要预测的图片
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) # 增加一个维度，batch通道数
    # [5] 归一化处理
    x = preprocess_input(x)
    # [6] 预测结果并输出
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))