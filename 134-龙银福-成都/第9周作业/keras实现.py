[1]
'''
加载训练数据和测试数据
打印训练数据集，训练标签，测试数据集，测试标签的shape
'''
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)

[2]
"""
打印测试集的第一张图片
"""
import matplotlib.pyplot as plt
plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.show()

[3]
'''
使用tensorflow.keras搭建神经网络
'''
from tensorflow.keras import models
from tensorflow.keras import layers

networks = models.Sequential()
networks.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
networks.add(layers.Dense(10, activation='softmax'))

networks.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

[4]
'''
数据预处理
把图像数据展平成一个向量
把标签转化成one_hot形式
'''
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
print(test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(test_labels[0])

[5]
'''
训练数据
'''
networks.fit(train_images, train_labels, epochs=5, batch_size=128)

[6]
'''
对训练好的模型进行测试
'''
test_loss, test_acc = networks.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print(test_acc)

[7]
'''
输入一张图片到模型，验证效果
'''
(train_images, train_labels), (test_images, test_lanels) = mnist.load_data()
plt.imshow(test_images[1], cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = networks.predict(test_images)

for i in range(res[1].shape[0]):
    if res[1][i] == 1:
        print(i)
        break