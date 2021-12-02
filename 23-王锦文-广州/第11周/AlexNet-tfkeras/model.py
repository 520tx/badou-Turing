#coding=utf-8
from __future__ import absolute_import,division
import tensorflow as tf
import numpy as np
import os
def build_AlexNet(input_shape=(224,224,3),output_shape=2):
    '''
    构建模型，为了加快速度，对kernel filters 数量减半了
    '''
    input_xs=tf.keras.layers.Input(shape=input_shape,name="input")
    out=tf.keras.layers.Conv2D(filters=48,kernel_size=(11,11),strides=(4,4),padding='same',activation='relu')(input_xs)
    out=tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='valid')(out)
    out=tf.keras.layers.Conv2D(filters=128,kernel_size=5,strides=1,padding='same',activation='relu')(out)
    out=tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='valid')(out)
    out=tf.keras.layers.Conv2D(filters=192,kernel_size=3,strides=1,padding='same',activation='relu')(out)
    out=tf.keras.layers.Conv2D(filters=192,kernel_size=3,strides=1,padding='same',activation='relu')(out)
    out=tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,padding='same',activation='relu')(out)
    out=tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='valid')(out)
    out=tf.keras.layers.Flatten()(out)
    out=tf.keras.layers.Dense(1024,activation='relu')(out)
    out=tf.keras.layers.Dropout(0.4)(out)
    out=tf.keras.layers.Dense(1024,activation='relu')(out)
    out=tf.keras.layers.Dropout(0.4)(out)
    out=tf.keras.layers.Dense(2)(out)
    logits=tf.keras.layers.Activation('softmax')(out)
    model=tf.keras.Model(input_xs,logits,name='output')
    print("finish build alexnet")
    return model


