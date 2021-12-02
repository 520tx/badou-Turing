from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# AlexNet 网络结构的实现
def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    model = Sequential()
    # 1-第一个卷积层：输出特征层(卷积核数量)为96层，卷积核大小为11x11，步长为4x4，填充方式为'valid'，激活函数为'relu'
    #   输入shape为(224, 224, 3), 输出shape为(55, 55, 96)
    #   此处为减少计算量，卷积核数量减半，即所建模型输出为 48 个特征层
    model.add(
        Conv2D(filters=48,
               kernel_size=(11, 11),
               strides=(4, 4),
               padding='valid',
               input_shape=input_shape,
               activation='relu'
        )
    )
    # 归一化层
    model.add(BatchNormalization())

    # 2-第一个最大池化层：核大小为 3x3，步长为 2x2(输出shape减半), 填充方式为'valid'
    #   输入shape为(55, 55, 96)，输出shape为(27, 27, 96)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 3-第二个卷积层：卷积核数量为256，核尺寸为5x5，步长为1x1
    #   输入shape为(27, 27, 96), 输出shape为(27, 27, 256)
    #   为减少计算量，此处卷积核数量减半，所建模型后输出为128个特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 归一化层
    model.add(BatchNormalization())

    # 4-第二个最大池化层：步长为2，shape减半
    #   输入shape为(27, 27, 256), 输出shape为(13, 13, 256)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 5-第三个卷积层：卷积核数量为384(输出特征层为384层)，卷积核大小为3x3，使用步长为1x1
    #   输入shape为(13, 13, 256)，输出shape为(13, 13, 384)
    #   为减少计算量，所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 6-第四个卷积层：卷积核数量384(输出特征层为384层),卷积核大小为3x3, 步长为1x1
    #   输入shape为(13, 13, 384)，输出的shape为(13, 13, 384)
    #   此处卷积核数量减半，所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 7-第五个卷积层：卷积核数量为256(输出特征层为256层), 卷积核大小为3x3, 步长为1x1
    #   输入shape为(13, 13, 384)，输出的shape为(13, 13, 256)
    #   为减少计算量，卷积核数量减半，所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )

    # 8-第三个最大池化层：步长为2x2(shape减半)
    #   输入shape为(13, 13, 256)，输出shape为(6, 6, 256)
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid'
        )
    )

    # 9-Flatten层
    #   输入shape为(6, 6, 256), 输出shape为(6*6*256 = 9216)
    #   此处为(6, 6, 128) --> 4608
    model.add(Flatten())

    # 10-全连接层
    #   输入 9216(缩减为4608)，输出 4096(缩减为1024)
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 11-全连接层
    #   输入 4096，输出 4096(缩减为1024)
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))

    # 12-全连接层
    #   输入 4096，输出 1000(这里改为2类)
    model.add(Dense(output_shape, activation='softmax'))

    return model