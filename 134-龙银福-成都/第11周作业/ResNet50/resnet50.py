#-------------------------------------------------------------#
#                    ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # [1] 输出通道数为filters1，卷积核大小1x1
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # [2] 输出通道数为filters2，卷积核大小3x3，填充方式：使得输入输出的 h 和 w 相等
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # [3] 输出通道数为filters3，卷积核大小1x1
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # [4] 求和
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # [1] 输出通道数为filters1，卷积核大小1x1，默认步长为2x2
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # [2] 输出通道数为filters2，卷积核大小3x3，填充方式：使得输入输出的 h 和 w 相等
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # [3] 输出通道数为filters3，卷积核大小1x1
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # [4] shortcut分支：输出通道数为filters3，卷积核大小1x1，默认步长为2x2
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    # [5] 求和
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):

    img_input = Input(shape=input_shape) # Input()：用来实例化一个keras张量，参数shape：形状元组(整型)，不包括batch size
    x = ZeroPadding2D(padding=(3, 3))(img_input) # 上下左右都补充3，行数加6，列数加6

    # [1] conv1：输出通道数64，卷积核7x7，步长为2
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # [2] conv2_x
    # conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2))
    # 输入张量为x，核尺寸3x3，conv_block中三个卷积层的输出通道数分别为64、64、256，步长为1
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    # identity_block(input_tensor, kernel_size, filters, stage, block)
    # 输入张量为x，核尺寸3x3，identity_block中三个卷积层的输出通道数分别为64、64、256
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # [3] conv3_x
    # conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2))
    # 输入张量为x，核尺寸3x3，conv_block中三个卷积层的输出通道数分别为128、128、256
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    # identity_block(input_tensor, kernel_size, filters, stage, block)
    # 输入张量为x，核尺寸3x3，identity_block中三个卷积层的输出通道数分别为128、128、512
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # [4] conv4_x
    # conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2))
    # 输入张量为x，核尺寸3x3，conv_block中三个卷积层的输出通道数分别为256、256、1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    # identity_block(input_tensor, kernel_size, filters, stage, block)
    # 输入张量为x，核尺寸3x3，identity_block中三个卷积层的输出通道数分别为256、256、1024
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # [5] conv5_x
    # conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2))
    # 输入张量为x，核尺寸3x3，conv_block中三个卷积层的输出通道数分别为512、512、2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    # identity_block(input_tensor, kernel_size, filters, stage, block)
    # 输入张量为x，核尺寸3x3，identity_block中三个卷积层的输出通道数分别为512、512、2048
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 池化窗口的大小(pool_height, pool_width)为7x7，输出为(1, 1, 2048)
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # 将输入的数据(展平)压成一维的数据
    x = Flatten()(x)
    # 构建一个有classes(1000)个节点的全连接层，激活函数为softmax
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    # 实例化模型：输入张量为img_input，输出张量为x
    model = Model(img_input, x, name='resnet50')

    # 加载权重
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':

    # [1] 实例化模型
    model = ResNet50()

    # [2] keras构建深度学习模型，可以通过model.summary()输出模型各层的参数状况
    model.summary()

    # [3] 加载需要预测的图像
    # img_path = 'elephant.jpg'
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # preprocess_input()是tensorflow下keras自带的类似于一个归一化的函数
    print('Input image shape:', x.shape)

    # [4] 预测
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))