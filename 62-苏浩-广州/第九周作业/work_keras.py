# 将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
from tensorflow.keras.datasets import mnist
# train_images是用于训练系统的手写数字图片;train_labels是用于标注图片的信息;
# test_images是用于检测系统训练效果的图片；test_labels是test_images图片对应的数字标签。
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

#把用于测试的第一张图片打印出来看看
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 使用tensorflow.Keras搭建一个有效识别图案的神经网络
# layers:表示神经网络中的一个数据处理层。(dense:全连接层)
from tensorflow.keras import models
from tensorflow.keras import layers
# models.Sequential():表示把每一个数据处理层串联起来.
network = models.Sequential()
# layers.Dense(…):构造一个数据处理层。input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# 归一化处理:
# reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，现在把每个二维数组转变为一个含有28*28个元素的一维数组.
train_images = train_images.reshape((60000, 28*28))
# train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 标记转化，例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0,]
from tensorflow.keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])


# 把数据输入网络进行训练：
# batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
# epochs:每次计算的循环是五次
network.fit(train_images, train_labels, epochs=5, batch_size = 128)


# 测试数据输入，检验网络学习后的图片识别效果.识别效果与硬件有关（CPU/GPU）.
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss) 
print('test_acc', test_acc)

# 输入一张手写数字图片到网络中，看它的识别效果
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break

