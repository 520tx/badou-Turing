#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
@Project ：badou-Turing 
@File    ：fmnist_tf_nn.py
@Author  ：luigi
@Date    ：2021/9/10 4:20 下午 
'''

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import tensorflow_datasets as tfds
from tensorflow import keras
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse


def get_data(image_path):
    keras.datasets.fashion_mnist.load_data()
    train_batch, val_batch = tfds.load(name="mnist", split=["train[:75%]", "train[:75%]"], data_dir=image_path,
                                       shuffle_files=True)
    return train_batch, val_batch


def get_model():
    model = Sequential()
    model.add(Dense(256, input_shape=(784,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = SGD(0.01)

    return model, optimizer


def train_batch(train, validation_data, model, optimizer):
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    train_x, train_y = train['image'], train['label']
    train_x = tf.reshape(train_x, [-1, 784])/255
    train_y = tf.one_hot(train_y, 10)

    val_x, val_y = validation_data['image'], validation_data['label']
    val_x = tf.reshape(val_x, [-1, 784])/255
    val_y = tf.one_hot(val_y, 10)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    H = model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=100, batch_size=32, callbacks=[callback])

    predictions = model.predict(val_x, batch_size=32)
    print(classification_report(val_y.numpy().argmax(axis=1), predictions.argmax(axis=1)))

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, 100), H.history["loss"], label='train_loss')
    plt.plot(np.arange(0, 100), H.history["val_loss"], label='val_loss')
    plt.plot(np.arange(0, 100), H.history["accuracy"], label='accuracy')
    plt.plot(np.arange(0, 100), H.history["val_accuracy"], label='val_accuracy')
    plt.title("training loss and accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="specify the dataset for mnist")
    ap.add_argument("-b", "--batch_size", default=32, help="batch_size")
    args = vars(ap.parse_args())

    train_set, test_set = get_data(args["path"])
    batch_size = args["batch_size"]
    model, optimizer = get_model()

    for batch_train, batch_val in zip(train_set.batch(batch_size), test_set.batch(batch_size)):
        train_batch(batch_train, batch_val, model, optimizer)


if __name__ == '__main__':
    main()
