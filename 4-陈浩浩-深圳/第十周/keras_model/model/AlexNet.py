from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout


def AlexNet(input_shape=(224, 224, 3), classes=1000, weight_file=None):
    #AlexNet论文：https://www.nvidia.cn/content/tesla/pdf/machine-learning/imagenet-classification-with-deep-convolutional-nn.pdf
    # 采用数据集：ImageNet LSVRC-2010
    # input_shape: 原始论文中对ImageNet(256, 256, 3)的图片，裁剪出中间(224, 224, 3)的图片
    # classes： ImagetNet共1000个分类

    model = Sequential()
    # 原始论文中使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 这里采用输出为48特征层
    model.add(Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                     activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())  # 原始论文中采用局部归一化，这里采用BacthNorm代替
    #原始论文使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 原始论文使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 这里采用输出为128特征层
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())  # 原始论文中采用局部归一化，这里采用BacthNorm代替
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 原始论文使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 这里采用输出为192特征层
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 原始论文使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 这里采用输出为192特征层
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 这里采用输出输出为128特征层
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    #原始论文使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))


    model.add(Flatten())
    # 原始论文中采用全连接层为4096， 这里采用1024
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.25))  # 原始论文采用0.5

    # 原始论文中采用全连接层为4096， 这里采用1024
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.25))  # 原始论文采用0.5

    model.add(Dense(units=classes, activation='softmax'))

    if weight_file:
        model.load_weights(weight_file)

    return model



if __name__ == "__main__":
    model = AlexNet()
    model.summary()



# torchvision ALexnet
# class AlexNet(nn.Module):
#
#     def __init__(self, num_classes=1000):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

