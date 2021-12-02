from keras.layers import Conv2D, MaxPooling2D, Input, Flatten,Dense
from keras.models import Model


# import keras.applications.vgg16
def Vgg16(input_shape=(224, 224, 3), classes=1000, weight_file=None):
    image_input = Input(shape=input_shape)   # 创建初始tensor

    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv1')(image_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(x)

    # block2
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(x)

    # block3
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(x)

    # block 4
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(x)

    # block 5
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # if include_top:
    #     # Classification block
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(4096, activation='relu', name='fc1')(x)
    #     x = Dense(4096, activation='relu', name='fc2')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    model = Model(inputs=image_input, outputs=x, name="vgg16")
    if weight_file:
        model.load_weights(weight_file)

    return model

