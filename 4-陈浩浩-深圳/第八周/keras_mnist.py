from keras.datasets import mnist
from keras import models, layers, utils
import keras
import matplotlib.pyplot as plt
import numpy as np


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, train_labels.shape)

# 图片预处理：resize,并进行归一化
train_images = train_images.reshape((-1, 28*28))
train_images = train_images.astype("float32")/255.0
test_images = test_images.reshape((-1, 28*28))
test_images = test_images.astype("float32")/255.0

# label预处理：one-hot编码
print("before change:", test_labels[0])
train_labels = utils.to_categorical(train_labels)
test_labels = utils.to_categorical(test_labels)
print("after change: ", test_labels[0])

# 模型搭建和训练
net = models.Sequential()
net.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
net.add(layers.Dense(10, activation="softmax"))

net.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss=keras.losses.categorical_crossentropy,
            metrics=["accuracy"])
# net.compile(optimizer="rmsprop", loss="categorical_crossentropy",
#             metrics=["accuracy"])

net.fit(train_images, train_labels, batch_size=128, epochs=5)  # 训练
test_loss, test_accu = net.evaluate(test_images, test_labels, verbose=1)
print("Test loss:{:.4f}, accuracy:{:.4f}%".format(test_loss, test_accu*100))


# 显示预测结果
for i in range(5):
    test_image = test_images[i:i+1, :]
    res = net.predict(test_image)
    print("Prediction digit: ", np.argmax(res, axis=1)[0])
    plt.imshow(test_image.reshape((28, 28)))
    plt.show()


