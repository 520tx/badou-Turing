from keras.layers import Conv2D, Dropout, BatchNormalization, \
    Input, Activation, Reshape, GlobalAveragePooling2D, ZeroPadding2D, DepthwiseConv2D
from keras.models import Model
from keras import backend as K


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """ conv2d+BN+Relu6

        # Arguments
            inputs: Input tensor of shape `(rows, cols, 3)`
            filters: Integer, the dimensionality of the output space
            alpha: controls the width of the network.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                     are used at each layer.
            kernel: An integer or tuple/list of 2 integers
            strides: An integer or tuple/list of 2 integers,
        """
    filters = int(filters*alpha)
    x = Conv2D(filters, kernel_size=kernel, padding='same',
               strides=strides, use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.

    # Arguments:

    pointwise_conv_filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the pointwise convolution).
    depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
    """
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', depth_multiplier=depth_multiplier,
                        strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


# import keras.applications.mobilenet
"""
1. 参数alpha
alpha:[1.0, 0.75, 0.5, 0.25], control the number of filters in each layer.
The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------

2. 输入图片尺寸
input_size = [224, 192, 160, 128], MobileNets support any input size greater than 32 x 32, with larger image sizes
offering better performance.
The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------
"""

def Mobilenet(input_shape=[224,224,3], alpha=1.0, depth_multiplier=1,
              dropout=1e-3,  classes=1000, weight_file=None):
    """
    alpha:[1.0, 0.75, 0.5, 0.25]
    """
    image_input = Input(shape=input_shape)
    # conv1_x:   shape(224, 224, 3)----->shape(112,112,64)
    x = _conv_block(image_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, strides=(1, 1), block_id=1)

    # conv2_x: shape(112, 112, 64)------>shape(56,56,128)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2,2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(1,1), block_id=3)

    # conv3_x: shape(56,56,128)------>shape(28,28,256)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2,2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(1,1), block_id=5)

    # conv4_x: shape(28, 28, 256)----->shape(14,14,512)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2,2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1,1), block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1,1), block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1,1), block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1,1), block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(1,1), block_id=11)

    # conv5_x: shape(14,14,512)------>shape(7,7, 1024)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(1, 1), block_id=13)

    # output:
    x = GlobalAveragePooling2D()(x)
    shape = (1, 1, int(1024*alpha))
    x = Reshape(target_shape=shape, name='reshape_1')(x)
    x = Dropout(rate=dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes, ), name='reshape_2')(x)

    # if include_top:
    #     if K.image_data_format() == 'channels_first':
    #         shape = (int(1024 * alpha), 1, 1)
    #     else:
    #         shape = (1, 1, int(1024 * alpha))
    #
    #     x = GlobalAveragePooling2D()(x)
    #     x = Reshape(shape, name='reshape_1')(x)
    #     x = Dropout(dropout, name='dropout')(x)
    #     x = Conv2D(classes, (1, 1),
    #                padding='same', name='conv_preds')(x)
    #     x = Activation('softmax', name='act_softmax')(x)
    #     x = Reshape((classes,), name='reshape_2')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    model = Model(inputs=image_input, outputs=x, name='mobilenet')
    if weight_file:
        model.load_weights(weight_file)
    return model

def my_preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == "__main__":
    from keras.preprocessing import image
    from keras.applications.imagenet_utils import preprocess_input, decode_predictions
    import numpy as np

    mobilenet = Mobilenet([224, 224, 3], alpha=1, depth_multiplier=1,
                          classes=1000, weight_file=r"../weight/mobilenet_1_0_224_tf.h5")
    mobilenet.summary()
    img = image.load_img("../weight/elephant.jpg", target_size=(224, 224))   # 读取图片，并resize到(224, 224)
    x = image.img_to_array(img)   # 转换PIL image格式为numpy array
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)  # 归一化：减均值，除标准差
    x = my_preprocess_input(x)  # 归一化：减均值，除标准差

    prediction = mobilenet.predict(x)
    print("Prediction: ", decode_predictions(prediction))   # 预测值，映射会label name