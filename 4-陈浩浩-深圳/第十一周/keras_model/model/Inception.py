from keras.layers import Conv2D, Dense, MaxPooling2D, BatchNormalization, \
    Input, Activation, concatenate, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model

def conv2d_bn(x, filters, num_row, num_col, padding='same',strides=(1, 1), name=None):
    """apply conv + BN + Relu.

        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.
    """
    if name is not None:
        conv_name = name+"_conv"
        bn_name = name+"_bn"
    else:
        conv_name = None
        bn_name = None
    x = Conv2D(filters, kernel_size=(num_row, num_col),
               strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=name)(x)   # scale: False时不乘gamma  (默认为：gamma*bn+beta)
    x = Activation('relu', name=name)(x)
    return x

# import keras.applications.inception_v3
def InceptionV3(input_shape=[224, 224, 3], classes=1000, weight_file=None):
    image_input = Input(shape=input_shape)
    # conv1_x
    x = conv2d_bn(image_input, filters=32, num_row=3, num_col=3, padding='valid', strides=(2, 2))
    x = conv2d_bn(x, filters=32, num_row=3, num_col=3, padding='valid', strides=(1, 1))
    x = conv2d_bn(x, filters=64, num_row=3, num_col=3, padding='same', strides=(1, 1))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # conv2_x
    x = conv2d_bn(x, filters=80, num_row=1, num_col=1, padding='valid', strides=(1, 1))
    x = conv2d_bn(x, filters=192, num_row=3, num_col=3, padding='valid', strides=(1, 1))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # conv3_x:     shape(35, 35)
    #Inception_3a:   shape(35, 35, 192)-----shape(35, 35, 256)
    branch1x1=conv2d_bn(x, 64, 1, 1, 'same', strides=(1, 1))

    branch5x5 = conv2d_bn(x, 48, 1, 1, 'same', (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, 'same', (1, 1))

    branch3x3dbl=conv2d_bn(x, 64, 1, 1, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'same', (1, 1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, 'same', (1, 1))

    # 64+64+96+32 = 256
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed0'
    )

    # Inception_3b:   shape(35, 35, 256)----->shape(35, 35, 288)
    branch1x1 = conv2d_bn(x, 64, 1, 1, 'same', (1, 1))

    branch5x5 = conv2d_bn(x, 48, 1, 1, 'same', (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, 'same', (1, 1))

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'same', (1, 1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, 'same', (1, 1))
    # 64+64+96+64 = 288
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed1'
    )

    # Inception_3c:  shape(35, 35, 288)----->shape(35, 35, 288)
    branch1x1 = conv2d_bn(x, 64, 1, 1, 'same', (1, 1))

    branch5x5 = conv2d_bn(x, 48, 1, 1, 'same', (1, 1))
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, 'same', (1, 1))

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'same', (1, 1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, 'same', (1, 1))
    # 64+64+96+64 = 288
    x = concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2'
    )

    # conv4_x :    shape(17, 17)
    # Inception_4a:    shape(35, 35, 288)----->shape(17, 17, 768)
    branch3x3 = conv2d_bn(x, 384, 3, 3, padding='valid', strides=(2, 2))

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'same', (1, 1))
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, 'valid', (2, 2))

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # 384+96+288 = 768
    x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed3'
    )

    # Inception_4b:    shape(17, 17, 768)----->shape(17, 17, 768)
    branch1x1 = conv2d_bn(x, 192, 1, 1, 'same', (1, 1))

    branch7x7 = conv2d_bn(x, 128, 1, 1, 'same', (1, 1))
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, 'same', (1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, 'same', (1, 1))

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, 'same', (1, 1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, 'same', (1, 1))
    # 192 + 192 + 192 + 192 = 768
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4'
    )

    # Inception_4c: shape(17, 17, 768)----->shape(17, 17, 768)
    # Inception_4d: shape(17, 17, 768)----->shape(17, 17, 768)
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, 'same', (1, 1))

        branch7x7 = conv2d_bn(x, 160, 1, 1, 'same', (1, 1))
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, 'same', (1, 1))
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, 'same', (1, 1))

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, 'same', (1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, 'same', (1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, 'same', (1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, 'same', (1, 1))
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, 'same', (1, 1))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, 'same', (1, 1))
        # 192+192+192+192 = 768
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed'+str(5+i)
        )

    # Inception_4e: shape(17, 17, 768)----->shape(17, 17, 768)
    branch1x1 = conv2d_bn(x, 192, 1, 1, 'same', (1, 1))

    branch7x7 = conv2d_bn(x, 192, 1, 1, 'same', (1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7, 'same', (1, 1))
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, 'same', (1, 1))

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, 'same', (1, 1))
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, 'same', (1, 1))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, 'same', (1, 1))
    # 192+192+192+192 = 768
    x = concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7'
    )

    # conv5_x :    shape(8, 8)
    # Inception_5a:    shape(17, 17, 768)----->shape(8, 8, 1280)
    branch3x3 = conv2d_bn(x, 192, 1, 1, 'same', (1, 1))
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, 'valid', (2, 2))

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, 'same', (1, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7, 'same', (1, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1, 'same', (1, 1))
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, 'valid', (2, 2))

    branch_pool = MaxPooling2D((3, 3), strides=(2,2))(x)
    # 320+192+768=1280
    x = concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=3,
        name="mixed8"
    )

    # Inception_5b: shape(8, 8, 1280)----->shape(8, 8, 2048)
    # Inception_5c: shape(8, 8, 2048)----->shape(8, 8, 2048)
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, 'same', (1, 1))

        branch3x3 = conv2d_bn(x, 384, 1, 1, 'same', (1, 1))
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3, 'same', (1, 1))
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1, 'same', (1, 1))
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=3, name="mixed9_"+str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, 'same', (1, 1))
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3, 'same', (1, 1))
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3, 'same', (1, 1))
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1, 'same', (1, 1))
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, 'same', (1, 1))
        # 320 + (384+384) + (384+384) +192=2048
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name="mixed"+str(9+i)
        )

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(units=classes, activation='softmax', name='predictions')(x)
    # if include_top:
    #     # Classification block
    #     x = GlobalAveragePooling2D(name='avg_pool')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    model = Model(inputs=image_input, outputs=x, name="inception_v3")
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
    inception3 = InceptionV3([224, 224, 3], 1000,
                             weight_file="../weight/inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    inception3.summary()

    img = image.load_img(r"../weight/elephant.jpg", target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    x = my_preprocess_input(x)

    prediction = inception3.predict(x)
    print("prediction: ", decode_predictions(prediction, top=1))