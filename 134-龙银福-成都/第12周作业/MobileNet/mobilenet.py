# -------------------------------------------------------------#
#     MobileNet的网络部分
# -------------------------------------------------------------#
import numpy as np

from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


# mobilenet网络模型
# depth_multiplier参数控制在深度可分离卷积的逐深度步骤中每个输入通道生成多少个输出通道
def MobileNet(input_shape=[224, 224, 3], depth_multiplier=1,
              dropout=1e-3, classes=1000):
    # Input()用于实例化 keras 张量
    img_input = Input(shape=input_shape)

    # [1] 标准卷积层：步长为2，输出feature map的大小减半
    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # [2] 第一个深度可分离卷积层
    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # [3] 第二个深度可分离卷积层
    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)

    # [4] 第三个深度可分离卷积层
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # [5] 第四个深度可分离卷积层
    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)

    # [6] 第五个深度可分离卷积层
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # [7] 第六个深度可分离卷积层
    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)

    # [8] 第七到第十一个深度可分离卷积层：包含五个深度可分离卷积
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # [9] 第十二个深度可分离卷积层
    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)

    # [10] 第十三个深度可分离卷积层
    # 7,7,1024 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # [11] 全局平均池化层
    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)

    # [12] 全连接层
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    # 实例化模型
    inputs = img_input
    model = Model(inputs, x, name='mobilenet_1_0_224_tf')

    # 加载预训练的权重
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model


# 标准卷积层：卷积 + 归一化 + relu6激活
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel, padding='same', use_bias=False,
               strides=strides, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


# 深度可分离卷积 (depthwise separable convolution)
def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    # [1] depthwise convolutional filters (逐深度卷积 --> 滤波)
    # depth_multiplier参数控制在逐深度卷积步骤中每个输入通道生成多少个输出通道
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier,
                        strides=strides, use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    # [2] pointwise convolutional filters (逐点1x1卷积 --> 组合)
    # pointwise_conv_filters 是1x1卷积核的数量，也是输出通道数
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False,
               strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


# 激活函数：大于6的部分全部等于6
def relu6(x):
    return K.relu(x, max_value=6)


# 归一化函数 [-1, 1]
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    # [1] 实例化mobilenet网络模型
    model = MobileNet(input_shape=(224, 224, 3))
    model.summary()     # 输出网络模型各层的参数状况

    # [2] 加载需要预测的图片
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)   # 增加一个维度(batch维)
    x = preprocess_input(x)         # 归一化处理
    print('Input image shape:', x.shape)

    # [3] 做预测并输出结果
    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1