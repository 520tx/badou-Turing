# from keras.models import Sequential
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten,\
    Input, ZeroPadding2D, Activation, add, AveragePooling2D, Dense


def identity_block(input_tensor, kernel_size, filters, stage, block):
    # shortcut中没有卷积层，直接进行相加
    """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filters=filters1, kernel_size=(1, 1), name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same',
               name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+"2b")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters3, kernel_size=(1, 1), name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    # 在shortcut中有一个卷积层，卷积进行升维或降维，方便相加
    """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn'+ str(stage) + block + '_branch'

    x = Conv2D(filters=filters1, kernel_size=(1, 1), strides=strides, 
			name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    # 注意这里的padding，kernel_size
    x = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, kernel_size=(1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 注意这里的strides
    shortcut = Conv2D(filters3, kernel_size=(1, 1), strides=strides, name=conv_name_base+'1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base+"1")(shortcut)

    x = add([x, shortcut])   # add进行tensor的求和运算，不是concatenate
    x = Activation('relu')(x)
    return x



# 参考：keras.applications.ResNet50
def Resnet50(input_shape=[224, 224, 3], classes=1000, weight_file=None):
    image_input = Input(shape=input_shape)   # 创建初始tensor

    # conv1
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(image_input)  # 图片四周补0
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)   # BN有一个axis参数，表示channel所在的axis，会在channel这个维度进行BN
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # conv2_x
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # conv3_x
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # conv4_x
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # conv5_x
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D(pool_size=(7, 7), name="avg_pool")(x)

    x = Flatten()(x)
    x = Dense(units=classes, activation="softmax", name='fc1000')(x)

    # if include_top:
    #     x = Flatten()(x)
    #     x = Dense(classes, activation='softmax', name='fc1000')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)


    model = Model(inputs=image_input, outputs=x, name="resnet50")
    if weight_file:
        model.load_weights(weight_file)

    return model


if __name__ == "__main__":
    from keras.preprocessing import image
    from keras.applications.imagenet_utils import preprocess_input, decode_predictions
    import os
    import numpy as np
    # src = r"./data"
    src = r"D:\BaiduNetdiskDownload\【11】图像识别\代码\resnet50_tf"
    model = Resnet50(weight_file=(os.path.join(src, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")))
    model.summary()
    img_path = os.path.join(src, 'elephant.jpg')
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # 读取图片，并resize到(224, 224)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # 归一化：减均值，除标准差

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))   # 预测值，映射会label name
