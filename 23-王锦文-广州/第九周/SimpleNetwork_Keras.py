#coding=utf-8
import tensorflow as tf
import numpy as np
import  os
'''
简单使用keras实现BP网络
'''
os.environ['CUDA_VISIBLE_DEVICES']=''
print("tf_verison:",tf.__version__)

def build_model():
    '''
    构建网络结构
    '''
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(200,activation='sigmoid',input_shape=(784,)))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    return  model
def load_data(path):
    '''
    读取数据集
    '''
    with open(path,'r') as f:
        data_list = f.readlines()
    all_inputs=[]
    all_targets=[]
    for line in data_list:
        values=line.split(",")
        inputs = (np.asfarray(values[1:]))/255.0 * 0.99 + 0.01
        targets=int(values[0])#标签
        all_inputs.append(inputs)
        all_targets.append(targets)
    all_inputs=np.array(all_inputs)
    all_targets=np.array(all_targets)
    all_targets= tf.keras.utils.to_categorical(all_targets, 10)#进行one-hot
    return all_inputs,all_targets #(100,784),(100,10)
def train():
    #读数数据
    traindata,trainlabels=load_data('./dataset/mnist_train.csv')
   # print("train:",traindata.shape,trainlabels.shape)
    valdata,vallabels=load_data("./dataset/mnist_test.csv")
    #构建网络
    model=build_model()
    #定义优优化器
    sgd = tf.keras.optimizers.SGD(lr=0.1, decay=5e-4, momentum=0.9)
    #编译模型，loss使用categorical_crossentropy sparse_categorical_crossentropy
    #如果使用sparse_categorical_crossentropy 标签不需要one-hot处理
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #训练
    model.fit(traindata,trainlabels,batch_size=4,epochs=5,)
    #测试
    loss, accuracy = model.evaluate(valdata,vallabels)
    print("testset loss={},acc={}".format(loss,accuracy))

if __name__=='__main__':
    train()




