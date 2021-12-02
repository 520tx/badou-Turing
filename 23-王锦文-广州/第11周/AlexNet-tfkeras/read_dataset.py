from typing import ValuesView
import tensorflow as tf
import numpy as np
import cv2
import os
classes={'cat':0,'dog':1}
def Writedataintotxt(rootpath):
    '''
    将训练集先写入到txt，方便后续使用
    '''
    files=os.listdir(rootpath)
    with open('dataset.txt','w') as f:
        for file in files:
            # print(file.split('.')[0])
            # print(type(classes[file.split('.')[0]]))
            f.write(file + " {}\n".format(classes[file.split('.')[0]]))
    print("write txt finish\n")
    
def generator(rootpath,lines,batch_size,inputsize=(224,224),num_classes=2):
    i = 0
    while 1:
        X_train=[]
        Y_train=[]
        for j in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name,label=lines[i].strip(' ').split(" ")
            image=cv2.imread(os.path.join(rootpath,name))
            if (image.shape[0]!=224 or image.shape[1]!=224):
                image=cv2.resize(image,inputsize)
            image=np.ascontiguousarray(image[:, :, ::-1])
            image=image.astype(np.float32)/255.
            X_train.append(image)
            Y_train.append(int(label))
            i=(i+1)%len(lines)#训练完一个epoch后，i重新置0
        X_train=np.array(X_train).reshape(-1,inputsize[0],inputsize[1],3)
        Y_train= tf.keras.utils.to_categorical(np.array(Y_train), num_classes)#进行one-hot
        yield(X_train,Y_train)#yield的函数相当于一个geerator对象
            

        
        
def get_dataset(datapath):
    with open(datapath,'r') as f:
        lines=f.read().splitlines()
    np.random.seed(304)
    np.random.shuffle(lines)
    #切分训练集和验证集
    num_val=int(len(lines)*0.1)
    num_train=len(lines)-num_val
    train_sample=lines[:num_train]
    val_sample=lines[num_train:]
    return train_sample,val_sample
    


    
    